import json
import os
import shutil
import sys

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
INPUT_SIZE = 512
MASK_SIZE = 128
NUM_CLASSES = 150

MIRROR = "https://hf-mirror.com"
REPO_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

WALKABLE_KEYS = ("road", "sidewalk", "walkway", "path", "floor", "flooring", "rug", "carpet", "grass", "field", "terrain")


def download_file(url, dst):
    r = requests.get(url, timeout=1800, stream=True)
    if r.status_code != 200:
        return False
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return True


def load_model():
    from transformers import SegformerForSemanticSegmentation

    local = os.path.join("hf_models", REPO_ID.replace("/", "__"))
    os.makedirs(local, exist_ok=True)
    base = f"{MIRROR}/{REPO_ID}/resolve/main"
    if not download_file(f"{base}/config.json", os.path.join(local, "config.json")):
        raise SystemExit("config download failed")
    if not download_file(f"{base}/model.safetensors", os.path.join(local, "model.safetensors")):
        if not download_file(f"{base}/pytorch_model.bin", os.path.join(local, "pytorch_model.bin")):
            raise SystemExit("weights download failed")
    return SegformerForSemanticSegmentation.from_pretrained(local).eval()


def get_test_image():
    for url in (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://ultralytics.com/images/bus.jpg",
    ):
        try:
            data = requests.get(url, timeout=30).content
            with open("bus.jpg", "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"download failed: {e}")
    return False


def pick_channel_order(model):
    import torch
    from PIL import Image

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    im = Image.open("bus.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None])

    id2label = model.config.id2label
    walk_names = [n for i, n in id2label.items() if any(k in n.lower().strip() for k in WALKABLE_KEYS)]
    name_to_id = {v: k for k, v in id2label.items()}
    walk_ids_t = [name_to_id[n] for n in walk_names]

    best = ("rgb", 0.0)
    with torch.no_grad():
        for order in ("rgb", "bgr"):
            x = arr if order == "rgb" else arr.flip(1)
            x = (x / 255.0 - mean) / std
            logits = model(pixel_values=x).logits
            mask = torch.argmax(logits[0], dim=0).numpy()
            bottom = mask[int(MASK_SIZE * 0.7):, :]
            frac = float(np.isin(bottom, walk_ids_t).mean())
            print(f"torch order={order}: bottom walkable = {frac:.3f}")
            if frac > best[1]:
                best = (order, frac)
    return best[0]


def export_onnx(model, order, path):
    import torch

    class Deploy(torch.nn.Module):
        def __init__(self, net, order):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.bgr = order == "bgr"

        def forward(self, images):
            x = images
            if self.bgr:
                x = torch.flip(x, dims=[1])
            x = (x / 255.0 - self.mean) / self.std
            logits = self.net(pixel_values=x).logits
            return torch.argmax(logits, dim=1)

    wrap = Deploy(model, order).eval()
    dummy = torch.randint(0, 256, (1, 3, INPUT_SIZE, INPUT_SIZE)).float()
    try:
        torch.onnx.export(wrap, dummy, path, opset_version=17, input_names=["images"], output_names=["class_ids"], dynamo=False)
    except TypeError:
        torch.onnx.export(wrap, dummy, path, opset_version=17, input_names=["images"], output_names=["class_ids"])
    print(f"fp32 onnx: {os.path.getsize(path) / 1e6:.1f} MB")


def quantize_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print(f"int8 onnx: {os.path.getsize(int8_path) / 1e6:.1f} MB")


def validate(path, walk_ids):
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"inputs: {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")
    im = Image.open("bus.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None]
    out = sess.run(None, {"images": x})[0]
    print(f"output: shape={out.shape} dtype={out.dtype}")
    mask = out[0]
    bottom = mask[int(MASK_SIZE * 0.7):, :]
    frac = float(np.isin(bottom, walk_ids).mean())
    print(f"bottom walkable = {frac:.3f}")
    return frac


def main():
    os.makedirs(ASSETS, exist_ok=True)
    if not get_test_image():
        raise SystemExit("no test image")
    model = load_model()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    walk_ids = sorted(
        i for i, name in id2label.items()
        if any(k in name.lower().strip() for k in WALKABLE_KEYS)
    )
    print(f"walkable ({len(walk_ids)}): {[(i, id2label[i]) for i in walk_ids]}")
    order = pick_channel_order(model)
    print(f"chosen order: {order}")

    export_onnx(model, order, "seg_fp32.onnx")
    quantize_int8("seg_fp32.onnx", "seg_int8.onnx")

    frac32 = validate("seg_fp32.onnx", walk_ids)
    frac8 = validate("seg_int8.onnx", walk_ids)
    final = "seg_int8.onnx" if frac8 > 0.10 and frac8 > frac32 - 0.05 else "seg_fp32.onnx"
    print(f"chosen: {final}")
    shutil.copyfile(final, os.path.join(ASSETS, "seg.onnx"))
    assert max(frac32, frac8) > 0.10, f"validation failed: {frac32}, {frac8}"

    spec = {
        "walkable": walk_ids,
        "labels": {str(i): id2label[i] for i in walk_ids},
        "all": {str(i): id2label[i] for i in id2label},
        "offset": 0,
        "norm": None,
    }
    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    print(f"FINAL MODEL: {final} ({os.path.getsize(final) / 1e6:.1f} MB, segformer-b0, {order})")


if __name__ == "__main__":
    main()
