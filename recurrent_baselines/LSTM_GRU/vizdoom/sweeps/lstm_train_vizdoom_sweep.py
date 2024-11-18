import os
import datetime
import wandb
import comet_ml
import argparse
import yaml
from torch.utils.data import DataLoader

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from recurrent_baselines.LSTM_GRU.vizdoom import lstm_trainer
from TMaze_new.TMaze_new_src.utils import set_seed, get_intro_vizdoom
from VizDoom.VizDoom_src.utils import get_dataset, batch_mean_and_std, ViZDoomIterDataset

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
os.environ["OMP_NUM_THREADS"] = "1" 

with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']
# os.environ['COMET_API_KEY'] = wandb_config['comet_ml_api']

with open("recurrent_baselines/LSTM_GRU/vizdoom/config_lstm.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# python3 recurrent_baselines/LSTM_GRU/vizdoom/lstm_train_vizdoom.py --model_mode 'LSTM' --ckpt_folder 'LSTM' --text 'LSTM'


def create_args():
    parser = argparse.ArgumentParser(description='RATE VizDoom trainer') 

    parser.add_argument('--model_mode',     type=str, default='LSTM',  help='Model training mode. Available variants: "DT, DTXL, RATE (Ours), RATEM (RMT)"')    
    parser.add_argument('--start_seed',     type=int, default=1,       help='Start seed')
    parser.add_argument('--end_seed',       type=int, default=1,       help='End seed')
    parser.add_argument('--ckpt_folder',    type=str, default='ckpt',  help='Checkpoints directory')
    parser.add_argument('--text',           type=str, default='',      help='Short text description of rouns group')

    return parser

if __name__ == '__main__':
    get_intro_vizdoom()
    
    args = create_args().parse_args()
    model_mode = args.model_mode
    start_seed = args.start_seed
    end_seed = args.end_seed
    ckpt_folder = args.ckpt_folder
    TEXT_DESCRIPTION = args.text

    config["model_mode"] = model_mode

    # RUN = 1
    for RUN in range(start_seed, end_seed+1):
        set_seed(RUN)
        print(f"Random seed set as {RUN}") 

        max_segments = config["training_config"]["sections"]

        """ MODEL MODE """

        if config["model_mode"] in ["LSTM", "GRU"]:
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]
            config['model_config']['arch_mode'] = config['model_mode'].lower()
            config['model_config']['max_length'] = 1#config["training_config"]["context_length"]

        print(f"Selected Model: {config['model_mode']}")  

        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace('-', '_')
        group = f'{TEXT_DESCRIPTION}_{config["model_mode"]}'
        name = f'{TEXT_DESCRIPTION}_{config["model_mode"]}_RUN_{RUN}_{date_time}'
        current_dir = os.getcwd()
        current_folder = os.path.basename(current_dir)
        ckpt_path = f'../{current_folder}/recurrent_baselines/LSTM_GRU/vizdoom/ckpt/{ckpt_folder}/{name}/'
        isExist = os.path.exists(ckpt_path)
        if not isExist:
            os.makedirs(ckpt_path)

        experiment = None
        if config["wandb_config"]["wwandb"]:
            # run = wandb.init(project=config['wandb_config']['project_name'], name=name, group=group, config=config, save_code=True, reinit=True)
            run = wandb.init(project=config['wandb_config']['project_name'], 
                config=config,
                save_code=True,
                reinit=True)
            sweep_config = wandb.config
            config["model_config"]["num_layers"] = sweep_config.get("model_config.num_layers", config["model_config"]["num_layers"])
            config["model_config"]["hidden_size"] = sweep_config.get("model_config.hidden_size", config["model_config"]["hidden_size"])
            config["model_config"]["dropout"] = sweep_config.get("model_config.dropout", config["model_config"]["dropout"])
        elif config['wandb_config']['wcomet']:
            print("*"*10, "COMET ML API SELECTED", "*"*10)

            comet_ml.init()

            experiment =comet_ml.Experiment(
                api_key=wandb_config['comet_ml_api'],
                project_name=config['wandb_config']['project_name'],
                workspace=wandb_config['workspace_name'],
                log_code=True
            )

            experiment.set_name(name)
            experiment.add_tags([group])
            experiment.log_parameters(config)
            

        #================================================== DATALOADERS CREATION ======================================================#
        
        # * IF USE ITER DATASET (5K TRAJECTORIES)
        # path_to_splitted_dataset = 'VizDoom/VizDoom_data/iterative_data/'
        path_to_splitted_dataset = '../../RATE/VizDoom/VizDoom_data/iterative_data/'
        train_dataset = ViZDoomIterDataset(path_to_splitted_dataset, 
                                         gamma=config["data_config"]["gamma"], 
                                         max_length=max_length, 
                                         normalize=config["data_config"]["normalize"])
        
        train_dataloader = DataLoader(train_dataset, 
                                     batch_size=config["training_config"]["batch_size"], 
                                     shuffle=True, 
                                     num_workers=8)

        print(f"Train: {len(train_dataloader) * config['training_config']['batch_size']} trajectories (first {max_length} steps)")

        if config["data_config"]["normalize"] == 0:
            type_norm = 'Without normalization'
        elif config["data_config"]["normalize"] == 1:
            type_norm = '/255.'
        elif config["data_config"]["normalize"] == 2:
            type_norm = 'Standard scaling'

        print(f'Normalization mode: {config["data_config"]["normalize"]}: {type_norm}')
        if config["data_config"]["normalize"] == 2:
            mean, std = batch_mean_and_std(train_dataloader)
            print("mean and std:", mean, std)
        else:
            mean, std = None, None
        #==============================================================================================================================#
        wandb_step = 0
        model = lstm_trainer.train(ckpt_path, config, train_dataloader, mean, std, max_segments, experiment)
                
        if config["wandb_config"]["wwandb"]:
            run.finish()
        elif config['wandb_config']['wcomet']:
            experiment.end()