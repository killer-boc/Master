# -*- coding: utf-8 -*-
"""OCR text detection on CanMV K230 (MicroPython + KPU).

Pipeline:
    sensor -> AI2D (letterbox pad + resize) -> PP-OCR text detection kmodel
    -> aicube.ocr_post_process (DB post-processing) -> quadrilaterals drawn on
    the OSD layer and printed to the serial console.

The script only needs the stock CanMV K230 firmware (``libs.*`` helpers,
``nncase_runtime`` and the built-in ``aicube`` module are all part of it).

Copy to the board (the kmodel ships with the firmware examples):
    /sdcard/examples/ocr_text_detect.py
    /sdcard/examples/kmodel/ocr_det_int16.kmodel

Run it from the CanMV IDE ("Run") or from the REPL:
    import ocr_text_detect
    ocr_text_detect.main()

Press Ctrl-C (or stop the script in the IDE) to release the sensor/display.
"""

import gc
import time

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import letterbox_pad_param
from media.media import ALIGN_UP

import aicube
import nncase_runtime as nn
import ulab.numpy as np

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# PP-OCR detection kmodel (shipped with the CanMV firmware examples).
DET_KMODEL_PATH = "/sdcard/examples/kmodel/ocr_det_int16.kmodel"

# Model input resolution [width, height] - must match the kmodel.
DET_INPUT_SIZE = [640, 640]

# Resolution handed to the AI task [width, height].
# Smaller is faster, larger detects smaller text.
RGB888P_SIZE = [640, 360]

# Display: "auto" picks the board default, or force e.g. "hdmi"/"lcd".
DISPLAY_MODE = "auto"
# None = use the display default resolution, or force e.g. [640, 480].
DISPLAY_SIZE = None

# DB post-processing thresholds:
#   mask_threshold - probability needed for a pixel to be "text"
#   box_threshold  - mean probability needed for a box to be kept
MASK_THRESHOLD = 0.25
BOX_THRESHOLD = 0.3

# Drop boxes smaller than this area (in AI-frame pixels) - removes noise.
MIN_BOX_AREA = 100

# Keep at most this many boxes per frame (0 = unlimited). Keeps the frame rate
# stable on busy scenes; the reading order is top-to-bottom, left-to-right.
MAX_BOXES = 20

# Box outline colour (ARGB) and thickness on the OSD image.
BOX_COLOR = (255, 255, 0, 0)
BOX_THICKNESS = 4

# Print the detected boxes every N frames (0 disables the log).
PRINT_INTERVAL = 30

# Print per-stage timings.
DEBUG_MODE = 0


# ---------------------------------------------------------------------------
# Geometry helpers (no hardware access, safe to reuse and test on a PC)
# ---------------------------------------------------------------------------

def clamp_quad(quad, width, height):
    """Convert an 8-value quad to ints clamped to the frame bounds."""
    out = []
    for i in range(0, 8, 2):
        x = int(quad[i])
        y = int(quad[i + 1])
        if x < 0:
            x = 0
        elif x > width - 1:
            x = width - 1
        if y < 0:
            y = 0
        elif y > height - 1:
            y = height - 1
        out.append(x)
        out.append(y)
    return out


def quad_area(quad):
    """Area of the quadrilateral given as 8 flat coordinates (shoelace)."""
    area = 0
    for i in range(4):
        x1 = quad[2 * i]
        y1 = quad[2 * i + 1]
        x2 = quad[(2 * i + 2) % 8]
        y2 = quad[(2 * i + 3) % 8]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def is_valid_box(quad, min_area=None):
    """True when the quad is big enough to be real text."""
    if min_area is None:
        min_area = MIN_BOX_AREA
    xs = (quad[0], quad[2], quad[4], quad[6])
    ys = (quad[1], quad[3], quad[5], quad[7])
    if max(xs) - min(xs) < 2 or max(ys) - min(ys) < 2:
        return False
    return quad_area(quad) >= min_area


def reading_order_key(item):
    """Sort key: text lines top-to-bottom, boxes inside a line left-to-right."""
    quad = item["quad"]
    ys = (quad[1], quad[3], quad[5], quad[7])
    xs = (quad[0], quad[2], quad[4], quad[6])
    line_height = max(ys) - min(ys)
    if line_height < 1:
        line_height = 1
    centre_y = (ys[0] + ys[1] + ys[2] + ys[3]) / 4.0
    return (int(centre_y / (line_height / 2.0)), min(xs))


def sort_results(results):
    """Order detection results for reading and cap the number of boxes."""
    ordered = sorted(results, key=reading_order_key)
    if MAX_BOXES > 0 and len(ordered) > MAX_BOXES:
        ordered = ordered[:MAX_BOXES]
    return ordered


def draw_results(osd_img, results, rgb888p_size, display_size,
                 color=BOX_COLOR, thickness=BOX_THICKNESS):
    """Draw every quad on the OSD image; returns the number of boxes drawn."""
    osd_img.clear()
    for item in results:
        quad = item["quad"]
        points = []
        for i in range(4):
            x = int(quad[2 * i] * display_size[0] / rgb888p_size[0])
            y = int(quad[2 * i + 1] * display_size[1] / rgb888p_size[1])
            points.append((x, y))
        for i in range(4):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % 4]
            osd_img.draw_line(x1, y1, x2, y2, color=color, thickness=thickness)
    return len(results)


def chw2hwc(features):
    """Copy a CHW array (sensor frame) into contiguous HWC layout."""
    shape = (features.shape[0], features.shape[1], features.shape[2])
    c_hw = features.reshape((shape[0], shape[1] * shape[2]))
    hw_c = c_hw.transpose()
    new_array = hw_c.copy()
    hwc = new_array.reshape((shape[1], shape[2], shape[0]))
    del c_hw
    del hw_c
    del new_array
    return hwc

# ---------------------------------------------------------------------------
# OCR text detection task
# ---------------------------------------------------------------------------

class OCRTextDetector(AIBase):
    """PP-OCR text region detector (DB) running on the K230 KPU."""

    def __init__(self, kmodel_path, model_input_size=None,
                 mask_threshold=MASK_THRESHOLD, box_threshold=BOX_THRESHOLD,
                 rgb888p_size=None, debug_mode=0):
        """Load the kmodel and prepare AI2D for NCHW uint8 frames.

        Args:
            kmodel_path (str): Path to the OCR detection kmodel.
            model_input_size (list or None): Model [width, height].
            mask_threshold (float): Pixel-level text probability threshold.
            box_threshold (float): Box-level score threshold.
            rgb888p_size (list or None): AI frame [width, height].
            debug_mode (int): Print timings when greater than zero.
        """
        model_input_size = list(model_input_size) if model_input_size else list(DET_INPUT_SIZE)
        rgb888p_size = list(rgb888p_size) if rgb888p_size else list(RGB888P_SIZE)
        # The sensor frame width must be aligned to 16 bytes.
        rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        try:
            self.kmodel_path = kmodel_path
            self.model_input_size = model_input_size
            self.mask_threshold = mask_threshold
            self.box_threshold = box_threshold
            self.rgb888p_size = rgb888p_size
            self.debug_mode = debug_mode
            self.ai2d = Ai2d(debug_mode)
            self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                     nn.ai2d_format.NCHW_FMT,
                                     np.uint8, np.uint8)
        except BaseException:
            # The caller cannot own this object when construction fails.
            try:
                self.deinit()
            except Exception as error:
                print("OCRTextDetector cleanup failed:", error)
            raise

    def config_preprocess(self, input_image_size=None):
        """Configure letterbox padding + bilinear resize for the kmodel."""
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
        top, bottom, left, right, _ = letterbox_pad_param(ai2d_input_size,
                                                          self.model_input_size)
        self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [0, 0, 0])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                        [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        """Decode the DB probability map into text boxes.

        Returns:
            list: ``[{"crop": <cropped text image>, "quad": [x0, y0, ...]}...]``
            sorted in reading order. ``crop`` is only used by the recognition
            script; the detection demo ignores it.
        """
        hwc_img = chw2hwc(self.cur_img)
        det_boxes = aicube.ocr_post_process(
            results[0][:, :, :, 0].reshape(-1),
            hwc_img.reshape(-1),
            self.model_input_size,
            self.rgb888p_size,
            self.mask_threshold,
            self.box_threshold,
        )
        del hwc_img
        detected = []
        for det_box in det_boxes:
            crop = det_box[0]
            quad = clamp_quad(det_box[1], self.rgb888p_size[0], self.rgb888p_size[1])
            if not is_valid_box(quad):
                continue
            detected.append({"crop": crop, "quad": quad})
        return sort_results(detected)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def main(max_frames=-1):
    """Run the live camera demo.

    Args:
        max_frames (int): Stop after this many frames; -1 runs until Ctrl-C.

    Returns:
        int: Number of processed frames.
    """
    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE,
                  display_size=DISPLAY_SIZE)
    detector = None
    created = False
    frames = 0
    started = time.time_ns()
    try:
        pl.create()
        created = True
        display_size = pl.get_display_size()
        detector = OCRTextDetector(DET_KMODEL_PATH,
                                   model_input_size=DET_INPUT_SIZE,
                                   mask_threshold=MASK_THRESHOLD,
                                   box_threshold=BOX_THRESHOLD,
                                   rgb888p_size=RGB888P_SIZE,
                                   debug_mode=DEBUG_MODE)
        detector.config_preprocess()
        while True:
            img = pl.get_frame()
            results = detector.run(img)
            draw_results(pl.osd_img, results, detector.rgb888p_size, display_size)
            pl.show_image()
            frames += 1
            if PRINT_INTERVAL > 0 and frames % PRINT_INTERVAL == 0:
                elapsed_ms = (time.time_ns() - started) / 1000000.0
                fps = frames * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
                print("[ocr_text_detect] frame=%d fps=%.1f boxes=%d"
                      % (frames, fps, len(results)))
                for item in results:
                    print("    quad:", item["quad"])
            gc.collect()
            if max_frames > 0 and frames >= max_frames:
                break
    finally:
        if detector is not None:
            try:
                detector.deinit()
            except Exception as error:
                print("OCRTextDetector deinit failed:", error)
        if created:
            try:
                pl.destroy()
            except Exception as error:
                print("PipeLine destroy failed:", error)
    return frames


if __name__ == "__main__":
    main()
