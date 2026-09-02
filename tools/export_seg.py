import json
import os
import shutil

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
INPUT_SIZE = 512
MASK_SIZE = 128
NUM_CLASSES = 150

CKPT_URL = "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_80k_ade20k/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth"
MIRROR = "https://hf-mirror.com"
LABELS_REPO = "nvidia/segformer-b1-finetuned-ade-512-512"

WALKABLE_KEYS = ("road", "sidewalk", "walkway", "path", "floor", "flooring", "rug", "carpet", "grass", "field", "terrain")


def download_ckpt():
    if os.path.exists("pspnet.pth"):
        return
    r = requests.get(CKPT_URL, timeout=1800)
    assert r.status_code == 200 and len(r.content) > 50000000, f"download failed: {r.status_code}"
    with open("pspnet.pth", "wb") as f:
        f.write(r.content)
    print(f"ckpt: {len(r.content) / 1e6:.1f} MB")


def load_labels():
    cfg = requests.get(f"{MIRROR}/{LABELS_REPO}/resolve/main/config.json", timeout=60).json()
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    assert len(id2label) == NUM_CLASSES
    return id2label


def build_pspnet():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet50

    class PSPHead(nn.Module):
        def __init__(self, in_ch=2048, mid=1024, out=512, ncls=NUM_CLASSES, bins=(1, 2, 3, 6)):
            super().__init__()
            self.stages = nn.ModuleList()
            for _ in bins:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(in_ch, mid, 1, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                ))
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_ch + mid * len(bins), out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            )
            self.conv_seg = nn.Conv2d(out, ncls, 1)

        def forward(self, x):
            h, w = x.shape[2:]
            priors = [x]
            for st in self.stages:
                y = F.adaptive_avg_pool2d(st(x), x.shape[2:])
                priors.append(y)
            feat = torch.cat(priors, dim=1)
            out = self.bottleneck(feat)
            out = F.dropout(out, 0.1, training=False)
            return self.conv_seg(out)

    class PSPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = resnet50(replace_stride_with_dilation=(False, True, True))
            self.decode_head = PSPHead()

        def forward(self, images):
            x = self.backbone.conv1(images)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            logits = self.decode_head(x)
            return F.interpolate(logits, size=(MASK_SIZE, MASK_SIZE), mode="bilinear", align_corners=False)

    net = PSPNet().eval()
    ckpt = torch.load("pspnet.pth", map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    mapped = {}
    for k, v in state.items():
        if k.startswith("decode_head."):
            if ".psp_modules." in k:
                nk = k.replace(".psp_modules.", ".stages.").replace(".conv.weight", ".0.weight").replace(".conv.bias", ".0.bias").replace(".bn.weight", ".1.weight").replace(".bn.bias", ".1.bias").replace(".bn.running_mean", ".1.running_mean").replace(".bn.running_var", ".1.running_var").replace(".bn.num_batches_tracked", ".1.num_batches_tracked")
                mapped[nk] = v
            elif ".bottleneck." in k:
                nk = k.replace(".conv.weight", ".0.weight").replace(".conv.bias", ".0.bias").replace(".bn.weight", ".1.weight").replace(".bn.bias", ".1.bias").replace(".bn.running_mean", ".1.running_mean").replace(".bn.running_var", ".1.running_var").replace(".bn.num_batches_tracked", ".1.num_batches_tracked")
                mapped[nk] = v
            elif k.startswith("decode_head.conv_seg."):
                mapped[k] = v
        elif k.startswith("backbone."):
            mapped[k] = v
    missing, unexpected = net.load_state_dict(mapped, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    real_missing = [m for m in missing if "num_batches_tracked" not in m and "fc." not in m]
    print(f"critical missing: {real_missing[:8]}")
    assert not real_missing, "state dict mapping incomplete"
    return net


def export_onnx(net, path):
    import torch

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    try:
        torch.onnx.export(net, dummy, path, opset_version=17, input_names=["images"], output_names=["logits"], dynamo=False)
    except TypeError:
        torch.onnx.export(net, dummy, path, opset_version=17, input_names=["images"], output_names=["logits"])
    print(f"onnx: {os.path.getsize(path) / 1e6:.1f} MB")


def validate(path, walk_ids):
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
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
        return "rgb", 0.0

    arr = np.asarray(im, dtype=np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    best = ("rgb", 0.0)
    for order in ("rgb", "bgr"):
        ch = arr if order == "rgb" else arr[:, :, ::-1]
        x = ((ch - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, {"images": x})[0]
        assert out.shape == (1, NUM_CLASSES, MASK_SIZE, MASK_SIZE), out.shape
        mask = np.argmax(out[0], axis=0)
        bottom = mask[int(MASK_SIZE * 0.7):, :]
        frac = float(np.isin(bottom, walk_ids).mean())
        print(f"order={order}: bottom walkable = {frac:.3f}")
        if frac > best[1]:
            best = (order, frac)
    return best


def main():
    os.makedirs(ASSETS, exist_ok=True)
    id2label = load_labels()
    walk_ids = sorted(
        i for i, name in id2label.items()
        if any(k in name.lower().strip() for k in WALKABLE_KEYS)
    )
    print(f"walkable ({len(walk_ids)}): {[(i, id2label[i]) for i in walk_ids]}")
    download_ckpt()
    net = build_pspnet()
    export_onnx(net, "seg.onnx")
    shutil.copyfile("seg.onnx", os.path.join(ASSETS, "seg.onnx"))

    order, frac = validate(os.path.join(ASSETS, "seg.onnx"), walk_ids)
    spec = {
        "walkable": walk_ids,
        "labels": {str(i): id2label[i] for i in walk_ids},
        "all": {str(i): id2label[i] for i in id2label},
        "offset": 0,
        "norm": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "order": order},
    }
    with open(os.path.join(ASSETS, "walkable.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    assert frac > 0.10, f"validation failed: {frac}"
    print(f"FINAL: PSPNet-R50 fp32, channel={order}, road-frac={frac:.3f}")


if __name__ == "__main__":
    main()
