import json
import os
import shutil

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
VOCAB = ["road", "sidewalk", "walkway", "path", "floor", "flooring", "rug", "carpet", "grass", "field", "terrain", "crosswalk"]
MODEL = "yoloe-11s-seg.pt"


def export():
    from ultralytics import YOLOE

    model = YOLOE(MODEL)
    pe = model.get_text_pe(VOCAB)
    model.set_classes(VOCAB, pe)
    out = model.export(format="onnx", imgsz=640)
    return str(out)


def validate(path):
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    im = None
    for url in (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://ultralytics.com/images/bus.jpg",
    ):
        try:
            data = requests.get(url, timeout=30).content
            with open("bus.jpg", "wb") as f:
                f.write(data)
            im = Image.open("bus.jpg").convert("RGB").resize((640, 640))
            break
        except Exception as e:
            print(f"download failed: {e}")
    if im is None:
        print("SKIP validation")
        return

    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    o0 = outs[0]
    nc = len(VOCAB)
    print(f"output0: {o0.shape}, output1: {outs[1].shape if len(outs) > 1 else None}")
    assert o0.shape[1] == 4 + nc + 32, f"unexpected channels: {o0.shape[1]}"
    road_best = float(o0[0, 4 + 0, :].max())
    sidewalk_best = float(o0[0, 4 + 1, :].max())
    print(f"road score: {road_best:.3f}, sidewalk score: {sidewalk_best:.3f}")
    assert road_best > 0.25 or sidewalk_best > 0.25, "no walkable surface detected on street test image"


def main():
    os.makedirs(ASSETS, exist_ok=True)
    src = export()
    print(f"exported: {src}")
    dst = os.path.join(ASSETS, "seg.onnx")
    shutil.copyfile(src, dst)
    spec = {
        "walkable": list(range(len(VOCAB))),
        "labels": {str(i): VOCAB[i] for i in range(len(VOCAB))},
        "all": {str(i): VOCAB[i] for i in range(len(VOCAB))},
    }
    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    validate(dst)
    print(f"FINAL MODEL: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB, {MODEL})")


if __name__ == "__main__":
    main()
