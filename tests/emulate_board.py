# -*- coding: utf-8 -*-
"""Offline smoke test for the CanMV K230 OCR scripts.

The board runtime (``libs.*``, ``nncase_runtime``, ``aicube``, ``media.*`` and
``ulab.numpy``) does not exist on a PC, so this module installs lightweight
stand-ins for those modules and runs the real scripts against them.

What is verified:
    * the scripts import and execute end to end (camera loop, drawing, cleanup);
    * AI2D build/run input and output shapes match the kmodel contract;
    * DB post-processing filters tiny boxes, clamps out-of-frame coordinates
      and sorts the boxes in reading order;
    * the CTC decoder merges repeats, drops blanks and produces the text;
    * tessellation of labels/boxes on the OSD image never gets non-int coords.

Run with:
    python3 tests/emulate_board.py
"""

import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Character set used by the fake recognition model (index 0 is unused padding,
# the last entry acts as the CTC blank because BLANK_INDEX is None).
FAKE_DICT_CHARS = ["_", "H", "E", "L", "O", " ", "X"]
# Timestep -> dictionary index produced by the fake recogniser logits:
# H, E, E, L, H, E, L, L  ->  after CTC merge  ->  "HELHEL"
FAKE_REC_SEQUENCE = [1, 2, 2, 3, 1, 2, 3, 3]
EXPECTED_TEXT = "HELHEL"

# Text boxes returned by the fake DB post-processing (8 flat coordinates).
FAKE_QUADS = [
    [40, 200, 240, 200, 240, 220, 40, 220],       # valid, second line
    [400, 120, 560, 120, 560, 140, 400, 140],     # valid, first line
    [10, 10, 14, 10, 14, 12, 10, 12],             # too small -> filtered
    [600, 340, 700, 340, 700, 380, 600, 380],     # clipped by frame bounds
]
EXPECTED_BOXES = 3

# ---------------------------------------------------------------------------
# Stub board modules
# ---------------------------------------------------------------------------


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    """Register the fake runtime modules (idempotent)."""
    if "libs" in sys.modules:
        return

    # ---- ulab.numpy -> the real numpy, API compatible for our usage ----
    ulab = _module("ulab")
    ulab.__path__ = []
    ulab_numpy = _module("ulab.numpy")
    for name in dir(np):
        if not name.startswith("_"):
            setattr(ulab_numpy, name, getattr(np, name))
    ulab.numpy = ulab_numpy

    # ---- nncase_runtime ----
    class _Ai2dFormat:
        NCHW_FMT = "nchw"
        RGB_packed = "rgb_packed"

    class _InterpMethod:
        tf_bilinear = "tf_bilinear"

    class _InterpMode:
        half_pixel = "half_pixel"

    class _Tensor:
        def __init__(self, array):
            self.array = array

        def to_numpy(self):
            return self.array

    class KPU:
        instances = []

        def __init__(self):
            self.path = ""
            self.outputs = []
            self.released = False
            KPU.instances.append(self)

        def load_kmodel(self, path):
            self.path = path

        def inputs_size(self):
            return 1

        def outputs_size(self):
            return 1

        def set_input_tensor(self, index, tensor):
            self.input = tensor

        def run(self):
            if "rec" in self.path:
                logits = np.full((1, len(FAKE_REC_SEQUENCE), len(FAKE_DICT_CHARS)), -10.0)
                for step, index in enumerate(FAKE_REC_SEQUENCE):
                    logits[0, step, index] = 10.0
                self.outputs = [_Tensor(logits)]
            else:
                self.outputs = [_Tensor(np.zeros((1, 40, 40, 1), dtype=np.float32))]

        def get_output_tensor(self, index):
            return self.outputs[index]

        def release(self):
            self.released = True

    def _shrink_memory_pool():
        return None

    _module("nncase_runtime",
            kpu=KPU,
            ai2d_format=_Ai2dFormat,
            interp_method=_InterpMethod,
            interp_mode=_InterpMode,
            shrink_memory_pool=_shrink_memory_pool)

    # ---- aicube (firmware DB post-processing) ----
    def ocr_post_process(feature_map, hwc_frame, model_input_size, rgb888p_size,
                         mask_threshold, box_threshold):
        assert len(feature_map) > 0, "empty probability map"
        assert hwc_frame is not None and len(hwc_frame) > 0, "empty frame data"
        boxes = []
        for quad in FAKE_QUADS:
            crop = np.zeros((1, 32, 128, 3), dtype=np.uint8)
            boxes.append([crop, list(quad)])
        return boxes

    _module("aicube", ocr_post_process=ocr_post_process)

    # ---- media (only ALIGN_UP is used by the scripts) ----
    def align_up(value, align):
        return (value + align - 1) // align * align

    media = _module("media")
    media.__path__ = []
    _module("media.media", ALIGN_UP=align_up)
    _module("media.sensor")
    _module("media.display", Display=object)

    # ---- image / ujson / utime, imported by libs.Utils ----
    class _Image:
        ARGB8888 = 0

        def __init__(self, *args, **kwargs):
            pass

    _module("image", Image=_Image, ARGB8888=0)
    _module("ujson")
    _module("utime", sleep_ms=lambda ms: None, ticks_ms=lambda: 0)

    # ---- libs.* helpers ----
    class ScopedTiming:
        def __init__(self, info="", enable_profile=True):
            self.info = info
            self.enable_profile = enable_profile

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    def letterbox_pad_param(input_size, output_size):
        """Same maths as the firmware helper in libs/Utils.py."""
        ratio_w = output_size[0] / input_size[0]
        ratio_h = output_size[1] / input_size[1]
        ratio = min(ratio_w, ratio_h)
        new_w = int(ratio * input_size[0])
        new_h = int(ratio * input_size[1])
        dw = (output_size[0] - new_w) / 2
        dh = (output_size[1] - new_h) / 2
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return top, bottom, left, right, ratio

    libs = _module("libs")
    libs.__path__ = []
    _module("libs.Utils", ScopedTiming=ScopedTiming,
            letterbox_pad_param=letterbox_pad_param)

    class Ai2d:
        """Records pre-processing configuration and validates shapes."""

        def __init__(self, debug_mode=0):
            self.debug_mode = debug_mode
            self.dtype = None
            self.in_shape = None
            self.out_shape = None
            self.pad_args = None

        def set_ai2d_dtype(self, in_fmt, out_fmt, in_dtype, out_dtype):
            self.dtype = (in_fmt, out_fmt, in_dtype, out_dtype)

        def pad(self, pads, pad_mode, value):
            self.pad_args = (list(pads), pad_mode, list(value))

        def resize(self, method, mode):
            pass

        def build(self, in_shape, out_shape):
            self.in_shape = tuple(in_shape)
            self.out_shape = tuple(out_shape)

        def run(self, input_np):
            if self.out_shape is None:
                raise RuntimeError("Ai2d.run() before build()")
            if self.in_shape is not None:
                got = tuple(input_np.shape)[-3:]
                want = self.in_shape[-3:]
                if got != want:
                    raise ValueError("AI2D input shape %s != build shape %s"
                                     % (got, want))
            return np.zeros(self.out_shape, dtype=np.uint8)

        def deinit(self):
            self.in_shape = None

    _module("libs.AI2D", Ai2d=Ai2d)

    # ---- libs.AIBase (simplified but contract-compatible) ----
    class AIBase:
        """Stand-in for the firmware AIBase: preprocess/inference/postprocess."""

        def __init__(self, kmodel_path, model_input_size=None,
                     rgb888p_size=None, debug_mode=0):
            self.kmodel_path = kmodel_path
            self.model_input_size = model_input_size
            self.rgb888p_size = rgb888p_size
            self.debug_mode = debug_mode
            self.cur_img = None
            self.tensors = []
            self.results = []
            self.kpu = sys.modules["nncase_runtime"].kpu()
            self.kpu.load_kmodel(kmodel_path)

        def preprocess(self, input_np):
            return [self.ai2d.run(input_np)]

        def inference(self, tensors):
            self.results.clear()
            for index, tensor in enumerate(tensors):
                self.kpu.set_input_tensor(index, tensor)
            self.kpu.run()
            for index in range(self.kpu.outputs_size()):
                self.results.append(self.kpu.get_output_tensor(index).to_numpy())
            return self.results

        def postprocess(self, results):
            return None

        def run(self, input_np):
            self.cur_img = input_np
            self.tensors = self.preprocess(input_np)
            self.results = self.inference(self.tensors)
            return self.postprocess(self.results)

        def deinit(self):
            if getattr(self, "_deinitialized", False):
                return
            self.cur_img = None
            kpu = getattr(self, "kpu", None)
            self.kpu = None
            if kpu is not None:
                kpu.release()
            if hasattr(self, "tensors"):
                self.tensors.clear()
                self.tensors = []
            if hasattr(self, "results"):
                self.results.clear()
                self.results = []
            ai2d = getattr(self, "ai2d", None)
            self.ai2d = None
            if ai2d is not None and hasattr(ai2d, "deinit"):
                ai2d.deinit()
            self._deinitialized = True

    _module("libs.AIBase", AIBase=AIBase)

    # ---- libs.PipeLine ----
    class OsdImage:
        """Records drawing calls and rejects invalid coordinates/text."""

        def __init__(self):
            self.calls = []

        def clear(self):
            self.calls.append(("clear",))

        def draw_line(self, x1, y1, x2, y2, color=None, thickness=1):
            for value in (x1, y1, x2, y2):
                if not isinstance(value, int):
                    raise TypeError("draw_line needs int coords, got %r" % (value,))
            if thickness < 1:
                raise ValueError("draw_line thickness must be >= 1")
            self.calls.append(("line", x1, y1, x2, y2))

        def draw_string_advanced(self, x, y, size, text, color=None, font=None):
            if not isinstance(x, int) or not isinstance(y, int):
                raise TypeError("draw_string_advanced needs int coords")
            if not isinstance(text, str) or not text:
                raise ValueError("draw_string_advanced needs a non-empty string")
            self.calls.append(("text", x, y, size, text, font))

    class PipeLine:
        instances = []

        def __init__(self, rgb888p_size=[224, 224], display_mode="hdmi",
                     display_size=None, osd_layer_num=1, debug_mode=0):
            self.rgb888p_size = [rgb888p_size[0], rgb888p_size[1]]
            self.display_size = [640, 480] if display_size is None else list(display_size)
            self.osd_img = OsdImage()
            self.created = False
            self.destroyed = False
            self.frames_served = 0
            PipeLine.instances.append(self)

        def create(self, sensor=None, sensor_id=None, hmirror=None, vflip=None,
                   fps=60, to_ide=True):
            self.created = True

        def get_frame(self):
            self.frames_served += 1
            return np.zeros((3, self.rgb888p_size[1], self.rgb888p_size[0]),
                            dtype=np.uint8)

        def get_display_size(self):
            return self.display_size

        def show_image(self, flag=None):
            pass

        def destroy(self):
            self.destroyed = True

    _module("libs.PipeLine", PipeLine=PipeLine, ScopedTiming=ScopedTiming)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _fresh():
    """Reset stub bookkeeping so every test observes only its own objects."""
    sys.modules["nncase_runtime"].kpu.instances = []
    sys.modules["libs.PipeLine"].PipeLine.instances = []


def _write_dict():
    fd, path = tempfile.mkstemp(suffix="_ocr_dict.txt")
    with os.fdopen(fd, "w", newline="") as f:
        f.write("\r\n".join(FAKE_DICT_CHARS))
    return path


def test_helpers():
    """Quad maths, filtering, clamping and reading order."""
    import ocr_text_detect as det_mod

    quad = [0, 0, 10, 0, 10, 10, 0, 10]
    assert det_mod.quad_area(quad) == 100.0
    assert det_mod.is_valid_box(quad)
    assert not det_mod.is_valid_box([0, 0, 3, 0, 3, 2, 0, 2])

    clamped = det_mod.clamp_quad([-5, -5, 700, 400, 700, 400, -5, -5], 640, 360)
    assert clamped[0] == 0 and clamped[1] == 0
    assert clamped[2] == 639 and clamped[3] == 359

    results = [
        {"quad": [40, 200, 240, 200, 240, 220, 40, 220]},
        {"quad": [400, 120, 560, 120, 560, 140, 400, 140]},
    ]
    assert det_mod.sort_results(results)[0]["quad"][0] == 400

    same_line = [
        {"quad": [300, 100, 400, 100, 400, 120, 300, 120]},
        {"quad": [10, 102, 100, 102, 100, 122, 10, 122]},
    ]
    assert det_mod.sort_results(same_line)[0]["quad"][0] == 10

    previous = det_mod.MAX_BOXES
    det_mod.MAX_BOXES = 1
    try:
        assert len(det_mod.sort_results(same_line)) == 1
    finally:
        det_mod.MAX_BOXES = previous


def test_detector_app():
    """Detector: AI2D shapes, box filtering/clipping, reading order."""
    import ocr_text_detect as det_mod

    detector = det_mod.OCRTextDetector("/tmp/ocr_det_int16.kmodel",
                                       model_input_size=[640, 640],
                                       rgb888p_size=[640, 360])
    try:
        detector.config_preprocess()
        assert detector.ai2d.in_shape == (1, 3, 360, 640), detector.ai2d.in_shape
        assert detector.ai2d.out_shape == (1, 3, 640, 640), detector.ai2d.out_shape

        frame = np.zeros((3, 360, 640), dtype=np.uint8)
        results = detector.run(frame)
        assert len(results) == EXPECTED_BOXES, len(results)
        assert [item["quad"][0] for item in results] == [400, 40, 600]
        for item in results:
            for value in item["quad"]:
                assert isinstance(value, int)
        clipped = results[2]["quad"]
        assert max(clipped[0::2]) <= 639
        assert max(clipped[1::2]) <= 359
        assert results[0]["crop"].shape == (1, 32, 128, 3)
    finally:
        detector.deinit()
    assert all(kpu.released for kpu in sys.modules["nncase_runtime"].kpu.instances)


def test_recognizer_app(dict_path):
    """Recogniser: dictionary parsing, AI2D shapes and CTC decoding."""
    import ocr_text_read as read_mod

    recognizer = read_mod.OCRTextRecognizer("/tmp/ocr_rec_int16.kmodel",
                                            model_input_size=[512, 32],
                                            dict_path=dict_path,
                                            rgb888p_size=[640, 360])
    try:
        assert recognizer.blank_index == len(FAKE_DICT_CHARS) - 1
        assert recognizer.dict_word[1] == "H"
        crop = np.zeros((1, 32, 128, 3), dtype=np.uint8)
        recognizer.config_preprocess(input_image_size=[128, 32], input_np=crop)
        assert recognizer.ai2d.in_shape == (1, 32, 128, 3), recognizer.ai2d.in_shape
        assert recognizer.ai2d.out_shape == (1, 3, 32, 512), recognizer.ai2d.out_shape
        assert recognizer.run(crop) == EXPECTED_TEXT
        # blanks are dropped and repeated indices merge
        assert recognizer.decode([6, 2, 2, 6, 3]) == "EL"
    finally:
        recognizer.deinit()
    assert all(kpu.released for kpu in sys.modules["nncase_runtime"].kpu.instances)


def test_detection_main():
    """Full camera loop of ocr_text_detect.py against the stub pipeline."""
    import ocr_text_detect as det_mod

    det_mod.DET_KMODEL_PATH = "/tmp/fake_ocr_det_int16.kmodel"
    det_mod.PRINT_INTERVAL = 1
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        frames = det_mod.main(max_frames=2)
    log = out.getvalue()
    assert frames == 2, frames
    assert "boxes=%d" % EXPECTED_BOXES in log, log
    assert "quad:" in log, log

    pipeline = sys.modules["libs.PipeLine"].PipeLine.instances[-1]
    assert pipeline.destroyed
    assert pipeline.frames_served == 2
    lines = [call for call in pipeline.osd_img.calls if call[0] == "line"]
    assert len(lines) == EXPECTED_BOXES * 4 * 2, len(lines)
    assert all(kpu.released for kpu in sys.modules["nncase_runtime"].kpu.instances)


def test_recognition_main(dict_path):
    """Full camera loop of ocr_text_read.py: boxes + recognised strings."""
    import ocr_text_read as read_mod

    read_mod.DET_KMODEL_PATH = "/tmp/fake_ocr_det_int16.kmodel"
    read_mod.REC_KMODEL_PATH = "/tmp/fake_ocr_rec_int16.kmodel"
    read_mod.DICT_PATH = dict_path
    read_mod.PRINT_INTERVAL = 1
    read_mod.OUTPUT_FILE = ""
    read_mod.FONT_PATH = "/tmp/SourceHanSansSC-Normal-Min.ttf"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        frames = read_mod.main(max_frames=1)
    log = out.getvalue()
    assert frames == 1, frames
    assert EXPECTED_TEXT in log, log
    assert "boxes=%d" % EXPECTED_BOXES in log, log

    pipeline = sys.modules["libs.PipeLine"].PipeLine.instances[-1]
    assert pipeline.destroyed
    labels = [call for call in pipeline.osd_img.calls if call[0] == "text"]
    assert len(labels) == EXPECTED_BOXES, len(labels)
    assert labels[0][4] == EXPECTED_TEXT, labels[0]
    assert labels[0][5] == read_mod.FONT_PATH, labels[0]
    assert all(kpu.released for kpu in sys.modules["nncase_runtime"].kpu.instances)


def main():
    _install_stubs()
    dict_path = _write_dict()
    try:
        _fresh()
        test_helpers()
        print("ok  geometry helpers")
        _fresh()
        test_detector_app()
        print("ok  detector")
        _fresh()
        test_recognizer_app(dict_path)
        print("ok  recognizer")
        _fresh()
        test_detection_main()
        print("ok  ocr_text_detect.main()")
        _fresh()
        test_recognition_main(dict_path)
        print("ok  ocr_text_read.main()")
    finally:
        os.remove(dict_path)
    print("all offline checks passed")


if __name__ == "__main__":
    main()
