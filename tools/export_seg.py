import json
import os
import shutil
import sys

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
CSAIL_DIR = "/tmp/csail"
INPUT_SIZE = 512
MASK_SIZE = 128
NUM_CLASSES = 150

BASE_HTTPS = "https://sceneparsing.csail.mit.edu/model/pytorch/ade20k-mobilenetv2dilated"
BASE_HTTP = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-mobilenetv2dilated"

WALKABLE_KEYS = ("road", "sidewalk", "walkway", "path", "floor", "flooring", "rug", "carpet", "grass", "field", "terrain")


def setup_code():
    os.system(f"git clone --depth 1 https://github.com/CSAILVision/semantic-segmentation-pytorch {CSAIL_DIR}")
    sys.path.insert(0, CSAIL_DIR)


def download_weights():
    for name in ("encoder_epoch_50.pth", "decoder_epoch_50.pth"):
        if os.path.exists(name):
            continue
        ok = False
        for base in (BASE_HTTPS, BASE_HTTP):
            try:
                r = requests.get(f"{base}/{name}", timeout=600)
                if r.status_code == 200 and len(r.content) > 1000000:
                    with open(name, "wb") as f:
                        f.write(r.content)
                    print(f"{name}: {len(r.content) / 1e6:.1f} MB")
                    ok = True
                    break
            except Exception as e:
                print(f"{base}/{name} failed: {e}")
        if not ok:
            raise SystemExit(f"cannot download {name}")


def load_labels():
    with open(os.path.join(CSAIL_DIR, "mit_semseg", "parser.py"), "r", errors="ignore") as f:
        pass
    import io
    import zipfile

    labels_url = "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/mit_semseg/utils/labels.py"
    text = None
    for u in (labels_url, f"https://gh-proxy.com/{labels_url}"):
        try:
            text = requests.get(u, timeout=30).text
            break
        except Exception as e:
            print(f"labels from {u} failed: {e}")
    if text is None:
        raise SystemExit("cannot download labels")
    start = text.index("object_index = [")
    end = text.index("]", start)
    seg = text[start + len("object_index = ["):end]
    names = [n.strip().strip("'\"") for n in seg.split(",") if n.strip()]
    print(f"object labels: {len(names)}, sample: {names[:5]}")
    assert len(names) == 150
    return {i: n for i, n in enumerate(names)}


def build_and_export():
    import torch

    from mit_semseg.models import ModelBuilder, SegmentationModule

    encoder = ModelBuilder.build_encoder(
        arch="mobilenetv2dilated", fc_dim=320, weights="encoder_epoch_50.pth"
    )
    decoder = ModelBuilder.build_decoder(
        arch="c1", fc_dim=320, num_class=NUM_CLASSES, weights="decoder_epoch_50.pth", use_softmax=True
    )
    net = SegmentationModule(encoder, decoder, None).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            return self.net(images, segSize=(MASK_SIZE, MASK_SIZE))

    wrap = Wrap(net)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    try:
        torch.onnx.export(wrap, dummy, "seg.onnx", opset_version=17, input_names=["images"], output_names=["logits"], dynamo=False)
    except TypeError:
        torch.onnx.export(wrap, dummy, "seg.onnx", opset_version=17, input_names=["images"], output_names=["logits"])
    print(f"onnx size: {os.path.getsize('seg.onnx') / 1e6:.1f} MB")


def validate(path, walk_ids):
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
            im = Image.open("bus.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
            break
        except Exception as e:
            print(f"download failed: {e}")
    if im is None:
        print("SKIP validation")
        return

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    x = (x - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    out = sess.run(None, {"images": x.astype(np.float32)})[0]
    print(f"logits shape: {out.shape}")
    assert out.shape == (1, NUM_CLASSES, MASK_SIZE, MASK_SIZE), f"unexpected shape {out.shape}"
    mask = np.argmax(out[0], axis=0)
    bottom = mask[int(MASK_SIZE * 0.7):, :]
    frac = float(np.isin(bottom, walk_ids).mean())
    allfrac = float(np.isin(mask, walk_ids).mean())
    print(f"bottom walkable fraction: {frac:.3f}, all: {allfrac:.3f}")
    assert frac > 0.10, f"validation failed: {frac:.3f}"


def main():
    os.makedirs(ASSETS, exist_ok=True)
    setup_code()
    download_weights()
    id2label = load_labels()
    walk_ids = sorted(
        i for i, name in id2label.items()
        if any(k in name.lower().strip() for k in WALKABLE_KEYS)
    )
    print(f"walkable ({len(walk_ids)}): {[(i, id2label[i]) for i in walk_ids]}")
    spec = {
        "walkable": walk_ids,
        "labels": {str(i): id2label[i] for i in walk_ids},
        "all": {str(i): id2label[i] for i in id2label},
        "offset": 0,
    }
    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    build_and_export()
    shutil.copyfile("seg.onnx", os.path.join(ASSETS, "seg.onnx"))
    validate(os.path.join(ASSETS, "seg.onnx"), walk_ids)
    print(f"FINAL MODEL: {os.path.getsize(os.path.join(ASSETS, 'seg.onnx')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
