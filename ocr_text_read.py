# -*- coding: utf-8 -*-
"""OCR text detection + recognition on CanMV K230 (MicroPython + KPU).

Two kmodels run per frame:
    1. ``ocr_det_int16.kmodel`` finds the text regions (see ocr_text_detect.py).
    2. ``ocr_rec_int16.kmodel`` reads every detected region and returns a string.

The text is drawn above each box and printed to the console (optionally written
to a file on the SD card).

Copy to the board:
    /sdcard/examples/ocr_text_read.py
    /sdcard/examples/ocr_text_detect.py          (imported, keep next to it)
    /sdcard/examples/kmodel/ocr_det_int16.kmodel
    /sdcard/examples/kmodel/ocr_rec_int16.kmodel
    /sdcard/examples/utils/dict.txt              (comes with the firmware)

Run it:
    import ocr_text_read
    ocr_text_read.main()
"""

import gc
import time

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import letterbox_pad_param
from media.media import ALIGN_UP

import nncase_runtime as nn
import ulab.numpy as np

from ocr_text_detect import (
    OCRTextDetector,
    draw_results,
    RGB888P_SIZE,
    DISPLAY_MODE,
    DISPLAY_SIZE,
    DET_KMODEL_PATH,
    DET_INPUT_SIZE,
    MASK_THRESHOLD,
    BOX_THRESHOLD,
    DEBUG_MODE,
)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# PP-OCR recognition kmodel and its input resolution [width, height].
REC_KMODEL_PATH = "/sdcard/examples/kmodel/ocr_rec_int16.kmodel"
REC_INPUT_SIZE = [512, 32]

# Character dictionary used by the recognition model.
DICT_PATH = "/sdcard/examples/utils/dict.txt"

# CTC blank token index. None = last dictionary entry (matches the official
# CanMV example). Set to 0 if the decoded text is shifted by one character.
BLANK_INDEX = None

# Print recognised text every N frames (0 disables the log).
PRINT_INTERVAL = 10

# Optional: overwrite this file with the text of the last frame ("" = disabled).
OUTPUT_FILE = ""

# Text label colour (ARGB) and font size on the OSD image.
TEXT_COLOR = (255, 0, 0, 255)
FONT_SIZE = 24

# Optional TTF used for the labels ("" = firmware default bitmap font).
# For Chinese text use e.g. "/sdcard/res/font/AlibabaPuHuiTi-3-45-Light.ttf".
FONT_PATH = ""

# ---------------------------------------------------------------------------
# OCR text recognition task
# ---------------------------------------------------------------------------

class OCRTextRecognizer(AIBase):
    """CTC text recogniser for a cropped text region."""

    def __init__(self, kmodel_path, model_input_size=None, dict_path=DICT_PATH,
                 rgb888p_size=None, debug_mode=0):
        """Load the recogniser and the character dictionary.

        Args:
            kmodel_path (str): Path to the OCR recognition kmodel.
            model_input_size (list or None): Model [width, height].
            dict_path (str): Path to the character dictionary file.
            rgb888p_size (list or None): Full AI frame [width, height].
            debug_mode (int): Print timings when greater than zero.
        """
        model_input_size = list(model_input_size) if model_input_size else list(REC_INPUT_SIZE)
        rgb888p_size = list(rgb888p_size) if rgb888p_size else list(RGB888P_SIZE)
        rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        try:
            self.kmodel_path = kmodel_path
            self.model_input_size = model_input_size
            self.rgb888p_size = rgb888p_size
            self.debug_mode = debug_mode
            self.dict_word = self.read_dict(dict_path)
            if not self.dict_word:
                raise ValueError("empty OCR dictionary: " + dict_path)
            if BLANK_INDEX is None:
                self.blank_index = len(self.dict_word) - 1
            else:
                self.blank_index = BLANK_INDEX
            self.ai2d = Ai2d(debug_mode)
            # The crop coming from the detector is packed RGB (NHWC).
            self.ai2d.set_ai2d_dtype(nn.ai2d_format.RGB_packed,
                                     nn.ai2d_format.NCHW_FMT,
                                     np.uint8, np.uint8)
        except BaseException:
            try:
                self.deinit()
            except Exception as error:
                print("OCRTextRecognizer cleanup failed:", error)
            raise

    def read_dict(self, dict_path):
        """Read the dictionary file into {index: character}.

        Args:
            dict_path (str): Path to the dictionary, one symbol per line.

        Returns:
            dict: Character index -> single character string.
        """
        with open(dict_path, "r") as f:
            text = f.read(200000)
        text = text.replace("\r\n", "\n")
        if text.endswith("\n"):
            text = text[:-1]
        lines = text.split("\n")
        words = {}
        for index, char in enumerate(lines):
            words[index] = char.replace("\r", "").replace("\n", "")
        return words

    def config_preprocess(self, input_image_size=None, input_np=None):
        """Configure letterbox padding + resize for one cropped region."""
        if input_np is None:
            raise ValueError("config_preprocess() needs input_np for the crop")
        if input_image_size is None:
            input_image_size = [input_np.shape[2], input_np.shape[1]]
        top, bottom, left, right, _ = letterbox_pad_param(input_image_size,
                                                          self.model_input_size)
        self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([input_np.shape[0], input_np.shape[1],
                         input_np.shape[2], input_np.shape[3]],
                        [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        """Greedy CTC decode of the recogniser logits."""
        preds = np.argmax(results[0], axis=2).reshape(-1)
        return self.decode(preds)

    def decode(self, preds):
        """Greedy CTC decoding: merge repeats, then drop blank tokens.

        Args:
            preds (ulab.numpy.ndarray): Per-timestep character indices.

        Returns:
            str: Recognised text.
        """
        text = ""
        previous = -1
        for i in range(len(preds)):
            index = int(preds[i])
            if index == previous:
                continue
            previous = index
            if index == self.blank_index:
                continue
            char = self.dict_word.get(index)
            if char:
                text += char
        return text

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def read_text(detector, recognizer, img):
    """Detect text boxes in ``img`` and read the text of every box.

    Args:
        detector (OCRTextDetector): Initialised detector.
        recognizer (OCRTextRecognizer): Initialised recogniser.
        img (ulab.numpy.ndarray): CHW sensor frame.

    Returns:
        list: ``[{"crop": ..., "quad": [...], "text": "..."}...]``.
    """
    results = detector.run(img)
    for item in results:
        crop = item["crop"]
        recognizer.config_preprocess(input_image_size=[crop.shape[2], crop.shape[1]],
                                     input_np=crop)
        item["text"] = recognizer.run(crop)
        # Free the crop and the intermediate tensors of this box right away.
        item["crop"] = None
        gc.collect()
    return results


def draw_text_results(osd_img, results, rgb888p_size, display_size,
                      text_color=TEXT_COLOR, font_size=FONT_SIZE,
                      font_path=FONT_PATH):
    """Draw the boxes plus the recognised string above each box."""
    draw_results(osd_img, results, rgb888p_size, display_size)
    for item in results:
        text = item.get("text")
        if not text:
            continue
        quad = item["quad"]
        xs = (quad[0], quad[2], quad[4], quad[6])
        ys = (quad[1], quad[3], quad[5], quad[7])
        x = int(min(xs) * display_size[0] / rgb888p_size[0])
        y = int(min(ys) * display_size[1] / rgb888p_size[1]) - font_size
        if y < 0:
            y = 0
        if font_path:
            osd_img.draw_string_advanced(x, y, font_size, text,
                                         color=text_color, font=font_path)
        else:
            osd_img.draw_string_advanced(x, y, font_size, text, color=text_color)



def results_to_text(results):
    """Join the recognised strings of one frame, one line per box."""
    lines = []
    for item in results:
        text = item.get("text")
        if text:
            lines.append(text)
    return "\n".join(lines)


def main(max_frames=-1):
    """Run the live camera OCR demo.

    Args:
        max_frames (int): Stop after this many frames; -1 runs until Ctrl-C.

    Returns:
        int: Number of processed frames.
    """
    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE,
                  display_size=DISPLAY_SIZE)
    detector = None
    recognizer = None
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
        recognizer = OCRTextRecognizer(REC_KMODEL_PATH,
                                       model_input_size=REC_INPUT_SIZE,
                                       dict_path=DICT_PATH,
                                       rgb888p_size=RGB888P_SIZE,
                                       debug_mode=DEBUG_MODE)
        while True:
            img = pl.get_frame()
            results = read_text(detector, recognizer, img)
            draw_text_results(pl.osd_img, results, detector.rgb888p_size, display_size,
                              font_path=FONT_PATH)
            pl.show_image()
            frames += 1
            text = results_to_text(results)
            if OUTPUT_FILE:
                with open(OUTPUT_FILE, "w") as f:
                    f.write(text + "\n")
            if PRINT_INTERVAL > 0 and frames % PRINT_INTERVAL == 0:
                elapsed_ms = (time.time_ns() - started) / 1000000.0
                fps = frames * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
                print("[ocr_text_read] frame=%d fps=%.1f boxes=%d"
                      % (frames, fps, len(results)))
                if text:
                    print(text)
            gc.collect()
            if max_frames > 0 and frames >= max_frames:
                break
    finally:
        if recognizer is not None:
            try:
                recognizer.deinit()
            except Exception as error:
                print("OCRTextRecognizer deinit failed:", error)
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
