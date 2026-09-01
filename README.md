# WalkCam 看路

手机本地离线的可通行区域分割 App。摄像头实时画面 + 绿色蒙版标注可通行区域 + 耗时/占比显示。

- 模型：SegFormer-B1 x2（Cityscapes 街道版 + ADE20K 室内版），int8 量化（MatMul）
- 推理：ONNX Runtime Android，纯 CPU，约 1 帧/秒
- 全程离线，无任何联网权限
- 底部按钮切换 室外/室内 模式
