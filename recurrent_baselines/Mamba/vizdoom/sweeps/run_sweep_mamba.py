import wandb
import yaml
import subprocess
import os

with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

with open("recurrent_baselines/Mamba/vizdoom/sweeps/sweep_config.yaml") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

with open("recurrent_baselines/Mamba/vizdoom/config_mamba.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

sweep_id = wandb.sweep(sweep_config, project=config['wandb_config']['project_name'])

def run_agent():
    subprocess.run(["python", "recurrent_baselines/Mamba/vizdoom/sweeps/mamba_train_vizdoom_sweep.py"])

wandb.agent(sweep_id, function=run_agent, count=50)