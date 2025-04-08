import pandas as pd
import os
import argparse
import math

def generate_popgym_command(env_name, env_index):
    csv_path = "run_experiments/POPGym/popgym_envs_info.csv"
    df = pd.read_csv(csv_path, sep=';')
    
    env_row = df[df['Environment'] == f'popgym-{env_name}-v0']
    
    if env_row.empty:
        raise ValueError(f"Environment popgym-{env_name}-v0 not found in CSV file")
    
    avg_episode_length = float(env_row['Average Episode Length'].values[0])
    act_dim = int(env_row['act_dim'].values[0])
    
    context_length = math.ceil(avg_episode_length / 3)
    max_length = context_length * 3
    
    command = f"""#!/bin/bash

python3 src/train.py \\
    --wandb.project-name='RATE-POPGym' \\
    --wandb.wwandb=False \\
    --data.gamma=1.0 \\
    --data.path-to-dataset='data/POPGym/popgym-{env_name}-v0/' \\
    --data.max-length={max_length} \\
    --training.learning-rate=0.0003 \\
    --training.lr-end-factor=0.1 \\
    --training.beta-1=0.9 \\
    --training.beta-2=0.999 \\
    --training.weight-decay=0.01 \\
    --training.batch-size=32 \\
    --training.warmup-steps=100 \\
    --training.final-tokens=10_000_000 \\
    --training.grad-norm-clip=5.0 \\
    --training.epochs=100 \\
    --training.ckpt-epoch=10 \\
    --training.online-inference=True \\
    --training.log-last-segment-loss-only=True \\
    --training.use-cosine-decay=True \\
    --training.context-length={context_length} \\
    --training.sections=3 \\
    --model.env-name='popgym-{env_name}' \\
    --model.state-dim=-1 \\
    --model.act-dim={act_dim} \\
    --model.n-layer=8 \\
    --model.n-head=10 \\
    --model.n-head-ca=2 \\
    --model.d-model=128 \\
    --model.d-head=256 \\
    --model.d-inner=64 \\
    --model.dropout=0.05 \\
    --model.dropatt=0.2 \\
    --model.mem-len=40 \\
    --model.ext-len=0 \\
    --model.num-mem-tokens=30 \\
    --model.mem-at-end=True \\
    --model.mrv-act='relu' \\
    --model.skip-dec-ffn=True \\
    --model.padding-idx=-10 \\
    --tensorboard-dir='runs/POPGym/{env_name}-v0' \\
    --model-mode='RATE' \\
    --arch-mode='TrXL' \\
    --start-seed=1 \\
    --end-seed=3 \\
    --text='' \\
    --online-inference.use-argmax=False \\
    --online-inference.episode-timeout=1001 \\
    --online-inference.desired-return-1=1.0
"""
    return command

def get_env_index(env_name):
    csv_path = "run_experiments/POPGym/popgym_envs_info.csv"
    df = pd.read_csv(csv_path, sep=';')
    
    env_full_name = f'popgym-{env_name}-v0'
    if env_full_name in df['Environment'].values:
        return df[df['Environment'] == env_full_name].index[0]
    else:
        raise ValueError(f"Environment {env_full_name} not found in CSV file")

def main():
    parser = argparse.ArgumentParser(description='Generate PopGym experiment script')
    parser.add_argument('--env', type=str, required=True, help='Environment name (e.g., MineSweeperEasy)')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    try:
        env_index = get_env_index(args.env)
        command = generate_popgym_command(args.env, env_index)
        
        if args.output:
            output_path = args.output
        else:
            output_path = f"run_experiments/run_{env_index}.sh"
        
        with open(output_path, 'w') as f:
            f.write(command)
        
        os.chmod(output_path, 0o755)
        
        print(f"Script successfully created: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()