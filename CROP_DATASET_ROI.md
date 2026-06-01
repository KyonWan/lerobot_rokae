# LeRobot 数据集图像裁剪工具

本工具用于对 LeRobot 数据集进行图像裁剪和尺寸调整，支持交互式 ROI 选择和批处理。

## 功能概述

- **交互式 ROI 选择**: 使用鼠标可视化选择裁剪区域
- **批量处理**: 一次性处理整个数据集的所有 episode
- **多相机支持**: 支持对多个相机视角分别设置裁剪参数
- **视频编码**: 支持 h264 等编码格式，加快处理速度
- **元数据保留**: 自动保留和更新任务描述、数据集版本等元数据

## 安装依赖

```bash
# 确保已安装 LeRobot 及其依赖
pip install lerobot

# 视频编解码依赖（推荐 h264 加速处理）
pip install torchcodec  # 或使用 --video-backend pyav

# 其他依赖
pip install opencv-python numpy tqdm
```

## 使用方法

### 1. 交互式裁剪（推荐首次使用）

运行脚本，会弹出 OpenCV 窗口让你用鼠标框选裁剪区域：

```bash
cd /path/to/lerobot

# 使用 PYTHONPATH 确保加载正确的 lerobot 模块
PYTHONPATH=/path/to/lerobot/src \
python lerobot/rl/crop_dataset_roi.py \
  --repo-id your_dataset \
  --root /path/to/your_dataset \
  --resize-height 480 \
  --resize-width 640 \
  --vcodec h264 \
  --video-backend torchcodec \
  --tolerance-s 0.05 \
  --task "你的任务描述"
```

**交互式操作说明：**

| 按键/操作 | 功能 |
|-----------|------|
| 鼠标左键拖拽 | 绘制矩形 ROI |
| `c` | 确认当前选择的区域 |
| `r` | 重置，重新选择 |
| `ESC` | 取消选择 |

### 2. 使用预定义裁剪参数

如果已经保存了裁剪参数文件（`crop_params.json`），可以直接使用：

```bash
PYTHONPATH=/path/to/lerobot/src \
python lerobot/rl/crop_dataset_roi.py \
  --repo-id your_dataset \
  --root /path/to/your_dataset \
  --crop-params-path /path/to/crop_params.json \
  --resize-height 480 \
  --resize-width 640 \
  --vcodec h264
```

**crop_params.json 格式示例：**

```json
{
    "observation.images.external": [240, 0, 240, 640],
    "observation.images.left_wrist": [0, 0, 480, 640]
}
```

格式说明：`[top, left, height, width]`

- `top`: 裁剪区域顶部 Y 坐标
- `left`: 裁剪区域左侧 X 坐标
- `height`: 裁剪区域高度
- `width`: 裁剪区域宽度

例如 `[240, 0, 240, 640]` 表示：
- 从第 240 行开始裁剪
- 从第 0 列开始裁剪
- 裁剪高度 240 像素
- 裁剪宽度 640 像素
- 最终得到下半部分图像

## 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--repo-id` | str | "lerobot" | 数据集仓库 ID |
| `--root` | str | None | 数据集根目录路径 |
| `--crop-params-path` | str | None | 裁剪参数 JSON 文件路径（可选） |
| `--new-repo-id` | str | None | 新数据集仓库 ID（默认原名称 + "_cropped_resized"） |
| `--resize-height` | int | 128 | 输出图像高度 |
| `--resize-width` | int | 128 | 输出图像宽度 |
| `--vcodec` | str | "h264" | 视频编码格式（推荐 h264 加速） |
| `--video-backend` | str | None | 视频解码后端（pyav / torchcodec） |
| `--tolerance-s` | float | 1e-4 | 时间戳容差（解决帧同步问题） |
| `--task` | str | "" | 自然语言任务描述 |
| `--push-to-hub` | flag | False | 是否推送到 HuggingFace Hub |

## 完整示例

### 示例 1：处理 external 相机下半部分

假设原始 external 相机分辨率为 480x640，只需要下半部分（240x640）：

```bash
PYTHONPATH=/home/rokae/code/wzy/lerobot_rokae/lerobot/src \
python /home/rokae/code/wzy/lerobot_rokae/lerobot/src/lerobot/rl/crop_dataset_roi.py \
  --repo-id classify_parts_05 \
  --root /home/rokae/code/wzy/lerobot_rokae/classify_parts_05 \
  --resize-height 480 \
  --resize-width 640 \
  --vcodec h264 \
  --video-backend torchcodec \
  --tolerance-s 0.05 \
  --task "将模块正确分类放置"
```

在弹出的窗口中，对 external 图像拖拽选择下半部分区域，然后按 `c` 确认。

### 示例 2：指定裁剪参数文件

```bash
# 1. 先手动创建裁剪参数文件
cat > /tmp/crop_params.json << 'EOF'
{
    "observation.images.external": [240, 0, 240, 640]
}
EOF

# 2. 使用参数文件运行
PYTHONPATH=/home/rokae/code/wzy/lerobot_rokae/lerobot/src \
python /home/rokae/Projects/lerobot_rokae/lerobot/src/lerobot/rl/crop_dataset_roi.py \
  --repo-id gripper_parts_single \
  --root /home/rokae/dataset/gripper_parts_single  \
  --crop-params-path /tmp/crop_params.json \
  --resize-height 480 \
  --resize-width 640 \
  --vcodec h264 \
  --new-repo-id gripper_parts_single_pi0
```

### 本仓库双臂录制数据（`scripts/record/rokae_record.sh --config_path=config/record/bi_rokae_spacemouse_example.yaml`，`--dataset.root` 为数据集根目录）

`--root` 必须与录制时 **`--dataset.root` 指向的目录**（内含 `meta/`）一致；`--repo-id` 与 **`--dataset.repo_id`** 一致。

**方式 A — 封装脚本（交互选 ROI）：**

```bash
export DATASET_ROOT=/home/rokae/dataset/gripper_parts_single   # 改成你本次路径
/home/rokae/Projects/lerobot_rokae/scripts/dataset/crop_bi_rokae_recorded.sh
```

**方式 B — 直接调用（交互选 ROI，无 JSON）：**

```bash
export PYTHONPATH=/home/rokae/Projects/lerobot_rokae/lerobot/src
export DATASET_ROOT=/home/rokae/classify_parts_test_20260108_153000
python /home/rokae/Projects/lerobot_rokae/lerobot/src/lerobot/rl/crop_dataset_roi.py \
  --repo-id test_2025/bi_rokae_record \
  --root "$DATASET_ROOT" \
  --resize-height 480 \
  --resize-width 640 \
  --vcodec h264 \
  --new-repo-id test_2025/bi_rokae_record_cropped \
  --tolerance-s 0.05
```

裁剪后的数据集会写在 **`$DATASET_ROOT` 的同级目录** 下（例如 `/home/rokae/bi_rokae_record_cropped`），不会覆盖原数据。

### 示例 3：保留原数据集，创建新数据集

```bash
PYTHONPATH=/home/rokae/code/wzy/lerobot_rokae/lerobot/src \
python /home/rokae/code/wzy/lerobot_rokae/lerobot/src/lerobot/rl/crop_dataset_roi.py \
  --repo-id classify_parts_05 \
  --root /home/rokae/code/wzy/lerobot_rokae/classify_parts_05 \
  --new-repo-id classify_parts_05_external_half \
  --resize-height 480 \
  --resize-width 640 \
  --vcodec h264
```

新数据集将保存到：`/home/rokae/code/wzy/lerobot_rokae/classify_parts_05_external_half`

## 常见问题

### 1. `torchvision.io` 没有 `VideoReader` 属性

**错误：** `AttributeError: module 'torchvision.io' has no attribute 'VideoReader'`

**解决：** 显式指定 `--video-backend torchcodec` 并确保已安装：

```bash
pip install torchcodec
```

### 2. 时间戳容差错误

**错误：** `AssertionError: queried timestamps unexpectedly violate the tolerance`

**解决：** 增加时间戳容差：

```bash
--tolerance-s 0.05  # 默认为 1e-4 (0.0001)，建议改为 0.05
```

### 3. 图像维度不匹配

**错误：** `ValueError: The feature 'observation.images.xxx' of shape '(3, 480, 640)' does not have the expected shape '(480, 640, 3)'`

**原因：** 图像格式应为 HWC `[H, W, C]`，但输入是 CHW `[C, H, W]`

**解决：** 脚本已自动处理格式转换，确保 `--resize-height` 和 `--resize-width` 与实际需求一致。

### 4. 输出目录已存在

**错误：** `FileExistsError: ... output_dir`

**解决：** 使用 `--new-repo-id` 指定新名称，或手动删除已存在的输出目录。

### 5. 模块导入错误

**错误：** `ImportError: cannot import name 'Empty' from partially initialized module 'queue'`

**解决：** 确保使用正确的 PYTHONPATH：

```bash
PYTHONPATH=/home/rokae/code/wzy/lerobot_rokae/lerobot/src python ...
```

## 输出说明

处理完成后，新数据集目录结构如下：

```
new_dataset_root/
├── meta/
│   ├── info.json          # 数据集元数据
│   ├── tasks.parquet      # 任务描述
│   ├── episodes/          # episode 元数据
│   └── crop_params.json   # 保存的裁剪参数
├── videos/                # 编码后的视频文件
│   ├── observation.images.external/
│   ├── observation.images.left_wrist/
│   └── observation.images.right_wrist/
└── ...
```

## Python API 使用

你也可以在 Python 代码中直接调用：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.crop_dataset_roi import (
    convert_lerobot_dataset_to_cropped_lerobot_dataset,
    get_image_from_lerobot_dataset,
)

# 加载原始数据集
dataset = LeRobotDataset(
    repo_id="classify_parts_05",
    root="/path/to/classify_parts_05",
    video_backend="torchcodec",
    tolerance_s=0.05,
)

# 定义裁剪参数
crop_params = {
    "observation.images.external": [240, 0, 240, 640],  # 下半部分
}

# 转换为新数据集
new_dataset = convert_lerobot_dataset_to_cropped_lerobot_dataset(
    original_dataset=dataset,
    crop_params_dict=crop_params,
    new_repo_id="classify_parts_05_cropped",
    new_dataset_root="/path/to/output",
    resize_size=(480, 640),
    vcodec="h264",
    task="将模块正确分类放置",
)
```

## 注意事项

1. **备份原始数据**：建议在处理前先备份原始数据集
2. **GPU 加速**：视频编码默认使用 CPU，大批量处理可能需要较长时间
3. **任务描述**：使用 `--task` 参数确保新数据集包含正确的自然语言指令
4. **版本兼容**：该工具适用于 LeRobot Dataset v3.0 格式

## 相关文件

- 主脚本：`lerobot/src/lerobot/rl/crop_dataset_roi.py`
- 数据集类：`lerobot/src/lerobot/datasets/lerobot_dataset.py`
