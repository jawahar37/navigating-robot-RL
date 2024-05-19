#!/bin/bash
#SBATCH --job-name=p3_w_gr        # Job name
#SBATCH -o training-%j.out     # Output file
#SBATCH -e training-%j.err     # Error file
#SBATCH --nodes=1              # Number of nodes
#SBATCH --tasks-per-node=1     # Number of tasks per node
#SBATCH --gres=gpu:4           # Number of GPUs required
#SBATCH --time=20:00:00        # Walltime
#SBATCH --mem=80G              # Memory per node in GB
#SBATCH -G 4

# Load necessary modules or activate environment (if required)
# module load cuda   # Load CUDA module if needed

# Command to execute your GPU program
export LD_LIBRARY_PATH=/common/home/jp2141/.conda/envs/rlgpu/lib:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

python3 main_navigation.py