#!/bin/bash
#SBATCH --job-name=eval_qa               # 任务名称
#SBATCH --cpus-per-task=8                # 每任务 CPU 核数
#SBATCH --gres=gpu:1                     # 申请1块GPU
#SBATCH --partition=titan                # 指定分区 (请根据您的集群修改)
#SBATCH --time=01:00:00                  # 预计运行时间1小时
#SBATCH --qos=titan                      # 服务质量 (请根据您的集群修改)
#SBATCH --output=eval_qa_%j.log          # 标准输出日志
#SBATCH --error=eval_qa_%j.err           # 标准错误日志

# --- 1. 环境与调试信息 ---
echo "========================================================"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================================"

# --- 2. 使用 Singularity Shell 和 Here Document ---
# 进入 Singularity Shell 并将后续命令作为输入
# 注意：这里我们假设您是从 /home/zhengf_lab/cse12210702 目录提交 sbatch 的
# 如果不是，请将下面的路径调整为绝对路径
singularity shell --nv --shell /bin/bash /home/zhengf_lab/cse12210702/cuda_11.3_sandbox <<EOF

# --- 步骤 2.1: 在容器内部激活 Conda 环境 ---
# 这一步非常关键
echo "--- Inside Singularity Container ---"
CONDA_BASE_PATH="/home/zhengf_lab/cse12210702/.conda" # <--- !! 请修改为您的Conda安装路径 !!
source "\$CONDA_BASE_PATH/etc/profile.d/conda.sh"

# --- 步骤 2.3: 定义文件路径并切换目录 ---
# 使用绝对路径确保脚本能找到文件
PROJECT_DIR="/home/zhengf_lab/cse12210702/3D-VisTA-0815"
PREDS_FILE="scanqa_result_250612.json"
GOLD_FILE="/home/zhengf_lab/cse12210702/ScannetData/annotations/qa/ScanQA_v1.0_val.json"

cd \$PROJECT_DIR
echo "Current directory inside container: \$(pwd)"

# --- 步骤 2.4: 运行评估脚本 ---
echo "--- Running Python Evaluation Script ---"
python evaluate_my_preds.py --preds_file \$PREDS_FILE --gold_file \$GOLD_FILE

# --- 步骤 2.5: 任务完成 ---
echo "--- Singularity Commands Finished ---"
EOF

# --- 3. Slurm 任务结束 ---
echo "========================================================"
echo "Slurm job finished at: $(date)"
echo "========================================================"