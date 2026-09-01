import json
import os
import shutil

import numpy as np
import requests

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
INPUT_SIZE = 512
MASK_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WALKABLE_KEYS = ("road", "sidewalk", "walkway", "path", "floor", "flooring", "rug", "carpet", "grass", "field", "terrain")

MODELS = [
    ("seg", "nvidia/segformer-b1-finetuned-ade-512-512", 150, True),
]

MIRROR = "https://hf-mirror.com"


def download_file(url, dst):
    r = requests.get(url, timeout=1800, stream=True)
    if r.status_code != 200:
        return False
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return True


def load_model(repo_id):
    from transformers import SegformerForSemanticSegmentation

    local = os.path.join("hf_models", repo_id.replace("/", "__"))
    os.makedirs(local, exist_ok=True)
    base = f"{MIRROR}/{repo_id}/resolve/main"
    if not download_file(f"{base}/config.json", os.path.join(local, "config.json")):
        raise RuntimeError(f"config.json download failed for {repo_id}")
    got = download_file(f"{base}/model.safetensors", os.path.join(local, "model.safetensors"))
    if not got:
        got = download_file(f"{base}/pytorch_model.bin", os.path.join(local, "pytorch_model.bin"))
    if not got:
        raise RuntimeError(f"weights download failed for {repo_id}")
    model = SegformerForSemanticSegmentation.from_pretrained(local).eval()
    return model


def walkable_ids(id2label):
    ids = []
    for i, name in id2label.items():
        low = name.lower().strip()
        if any(k in low for k in WALKABLE_KEYS):
            ids.append(int(i))
    return sorted(set(ids))


def export_onnx(model, path):
    import torch

    class LogitsOnly(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            return self.net(pixel_values=images).logits

    wrap = LogitsOnly(model)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    kwargs = dict(opset_version=17, input_names=["images"], output_names=["logits"])
    try:
        torch.onnx.export(wrap, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrap, dummy, path, **kwargs)
    try:
        import onnx
        import onnxsim

        m, ok = onnxsim.simplify(path)
        if ok:
            onnx.save(m, path)
            print("onnxsim ok")
    except Exception as e:
        print(f"onnxsim skipped: {e}")


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


def predict_mask(path):
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    im = Image.open("bus.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    x = (x - MEAN.reshape(1, 3, 1, 1)) / STD.reshape(1, 3, 1, 1)
    logits = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    if logits.shape[2] != MASK_SIZE:
        print(f"note: mask size {logits.shape[2]}")
    return np.argmax(logits[0], axis=0).astype(np.int32), logits.shape


def walkable_fraction(mask, walk_ids):
    walk = np.isin(mask, walk_ids)
    bottom = walk[int(mask.shape[0] * 0.7):, :]
    return float(bottom.mean()), float(walk.mean())


def main():
    os.makedirs(ASSETS, exist_ok=True)
    spec = {}
    if not get_test_image():
        print("WARNING: no test image, validation skipped")

    for key, repo_id, ncls, hard in MODELS:
        print(f"=== {key}: {repo_id}")
        model = load_model(repo_id)
        id2label = model.config.id2label
        wids = walkable_ids(id2label)
        print(f"walkable classes ({len(wids)}): {[(i, id2label[i]) for i in wids]}")
        spec = {
            "walkable": wids,
            "labels": {str(i): id2label[i] for i in wids},
        }

        fp32 = f"seg_{key}_fp32.onnx"
        export_onnx(model, fp32)
        print(f"fp32 size: {os.path.getsize(fp32) / 1e6:.1f} MB")
        del model

        if os.path.exists("bus.jpg"):
            mask, shape = predict_mask(fp32)
            frac_bottom, frac_all = walkable_fraction(mask, wids)
            print(f"fp32 mask shape={shape}, walkable bottom={frac_bottom:.2f}, all={frac_all:.2f}")
            if hard:
                assert frac_bottom > 0.15, f"{key}: road not detected at bottom ({frac_bottom:.2f})"

        final = fp32
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8 = f"seg_{key}_int8.onnx"
            quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
            print(f"int8 size: {os.path.getsize(int8) / 1e6:.1f} MB")
            if os.path.exists("bus.jpg"):
                m32, _ = predict_mask(fp32)
                m8, _ = predict_mask(int8)
                agree = float((m32 == m8).mean())
                print(f"int8 agreement vs fp32: {agree:.3f}")
                if agree < 0.90:
                    print("int8 degraded too much, keeping fp32")
                else:
                    final = int8
        except Exception as e:
            print(f"quantization skipped: {e}")

        shutil.copyfile(final, os.path.join(ASSETS, "seg.onnx"))
        print(f"FINAL {key}: {final}")

    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    print("walkable.json written")


if __name__ == "__main__":
    main()
