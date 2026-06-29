#!/bin/bash
# 录制入口（单臂 / 双臂相同）：可选绑核 + 调用 `lerobot_record`，具体机器与任务参数全部由命令行或 --config_path 提供。
#
# 多组参数推荐两种方式（二选一即可）：
#   1) draccus 官方：把一组参数放进 yaml/json，用 --config_path=...，后面仍可追加覆盖项。
#      仓库内示例: config/record/single_rokae_spacemouse_example.yaml
#                config/record/bi_rokae_spacemouse_example.yaml
#      ./scripts/record/rokae_record.sh --config_path=config/record/single_rokae_spacemouse_example.yaml
#      ./scripts/record/rokae_record.sh --config_path=config/record/single_rokae_spacemouse_example.yaml --dataset.root="./dataset/run_$(date +%Y%m%d_%H%M%S)"
#   2) 多个一行脚本：每个文件里写一条完整的 python ...（未入库的 .sh 可放任意目录并加入 .gitignore）。
#
# 需要绑 CPU 时（与旧脚本 taskset -c 0 类似）：
#   export RECORD_TASKSET_CPUS=0
#   ./scripts/record/rokae_record.sh --config_path=...
#
# 恢复录制：在参数里加 --resume=true，并把 --dataset.root 指到已有数据集目录等（与 LeRobot 文档一致）。
export RECORD_TASKSET_CPUS=4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

if [[ -n "${RECORD_TASKSET_CPUS:-}" ]]; then
    exec taskset -c "$RECORD_TASKSET_CPUS" python -m lerobot.scripts.lerobot_record "$@"
else
    exec python -m lerobot.scripts.lerobot_record "$@"
fi
