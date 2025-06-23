#!/bin/bash
#SBATCH --job-name=cls_nmn               # 任务名称
#SBATCH --cpus-per-task=24                  # 每任务 CPU 数
#SBATCH --gres=gpu:1                        # 只用单卡
#SBATCH --partition=titan                   # 指定分区
#SBATCH --time=24:00:00                     # 运行时间 48 小时
#SBATCH --qos=titan                          # 服务质量
#SBATCH --output=cls_nmn_%j.log          # 输出日志文件
#SBATCH --error=cls_nmn_%j.err           # 错误日志文件

echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# 切换到项目目录
cd /home/zhengf_lab/cse12210702/3D-VisTA-0815

# 激活conda环境
source ~/.bashrc
conda activate 3dvista

echo "which python: $(which python)"
echo "python version: $(python --version)"
echo "conda env: $CONDA_DEFAULT_ENV"
echo "CUDA version (from nvcc): $(nvcc --version || echo 'nvcc not found')"
python -c "import sys; print('Python sys.path:', sys.path)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# 运行主命令
python -u run.py --config project/vista/scanqa_eval.yml

echo "End time: $(date)"