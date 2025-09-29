#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --output=ppo_train_%j.out
#SBATCH --error=ppo_train_%j.err
#SBATCH --partition=gpu            # set your GPU partition
#SBATCH --gres=gpu:1              # number of GPUs you want
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                 # memory as needed
#SBATCH --time=24:00:00

# Load modules or activate environment
module load cuda/11.8            # adjust to your cluster
module load python/3.11

# Activate your Python venv if needed
source ~/envs/ms3-lerobot/bin/activate


export CUDA_LAUNCH_BLOCKING=1
export seed=3

python -m lerobot_sim2real.scripts.train_ppo_rgb --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
  --ppo.seed=$seed \
  --ppo.num_envs=1024 --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=32 \
  --ppo.total_timesteps=100000000 \
  --ppo.gamma=0.9 \
  --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
  --ppo.exp-name="ppo-SO100GraspCube-v1-rgb-$seed" \
  --ppo.track --ppo.wandb_project_name "SO100-ManiSkill"
