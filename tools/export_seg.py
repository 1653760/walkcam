import io
import json
import os
import shutil
import tarfile

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
INPUT_SIZE = 257
NUM_CLASSES_TFLITE = 151
MIRROR = "https://hf-mirror.com"
LABELS_REPO = "nvidia/segformer-b1-finetuned-ade-512-512"

MODEL_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_mnv2_dm05_ade20k_float.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/deeplabv3_mnv2_dm05_ade20k_float.tgz",
]

WALKABLE_KEYS = ("road", "sidewalk", "walkway", "path", "floor", "flooring", "rug", "carpet", "grass", "field", "terrain")


def download_model():
    for url in MODEL_URLS:
        try:
            print(f"downloading {url}")
            data = requests.get(url, timeout=600).content
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                names = tar.getnames()
                print(f"tar contents: {names}")
                target = None
                for n in names:
                    if n.endswith(".tflite"):
                        target = n
                        break
                if target is None:
                    continue
                tar.extract(target)
                return target
        except Exception as e:
            print(f"failed: {e}")
    raise SystemExit("no deeplab tflite model downloaded")


def load_labels():
    cfg = requests.get(f"{MIRROR}/{LABELS_REPO}/resolve/main/config.json", timeout=60).json()
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    print(f"labels: {len(id2label)} classes")
    return id2label


def main():
    os.makedirs(ASSETS, exist_ok=True)
    id2label = load_labels()
    walk_ids = sorted(
        i for i, name in id2label.items()
        if any(k in name.lower().strip() for k in WALKABLE_KEYS)
    )
    print(f"walkable ({len(walk_ids)}): {[(i, id2label[i]) for i in walk_ids]}")

    tflite_path = download_model()
    dst = os.path.join(ASSETS, "deeplab_ade.tflite")
    shutil.move(tflite_path, dst)
    print(f"model size: {os.path.getsize(dst) / 1e6:.1f} MB")

    spec = {
        "walkable": walk_ids,
        "labels": {str(i): id2label[i] for i in walk_ids},
        "all": {str(i): id2label[i] for i in id2label},
        "offset": 1,
    }
    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)

    validate(dst, walk_ids)


def validate(path, walk_ids):
    from PIL import Image
    from tflite_runtime.interpreter import Interpreter

    im = None
    for url in (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://ultralytics.com/images/bus.jpg",
    ):
        try:
            data = requests.get(url, timeout=30).content
            with open("bus.jpg", "wb") as f:
                f.write(data)
            im = Image.open("bus.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
            break
        except Exception as e:
            print(f"download failed: {e}")
    if im is None:
        print("SKIP validation")
        return

    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    details_in = interpreter.get_input_details()[0]
    details_out = interpreter.get_output_details()[0]
    print(f"input: {details_in['shape']} {details_in['dtype']}")
    print(f"output: {details_out['shape']} {details_out['dtype']}")

    arr = np.asarray(im, dtype=np.float32)
    tfl_ids = [w + 1 for w in walk_ids]

    best = None
    for name, prep in (("div255", arr / 255.0), ("mobilenet", arr / 127.5 - 1.0), ("raw", arr)):
        x = prep.astype(np.float32)[None]
        interpreter.set_tensor(details_in["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(details_out["index"])
        if out.shape[-1] == 1:
            mask = out[0, :, :, 0].astype(np.int32)
        else:
            mask = np.argmax(out[0], axis=-1)
        bottom = mask[int(INPUT_SIZE * 0.7):, :]
        frac = float(np.isin(bottom, tfl_ids).mean())
        print(f"prep={name}: bottom walkable fraction = {frac:.3f}")
        if best is None or frac > best[1]:
            best = (name, frac)

    print(f"BEST PREP: {best[0]} ({best[1]:.3f})")
    with open(os.path.join(ASSETS, "walkable.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)
    spec["input_prep"] = best[0]
    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    assert best[1] > 0.10, f"validation failed: best road fraction {best[1]:.3f}"


if __name__ == "__main__":
    main()
