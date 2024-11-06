import os
import datetime
import torch
import wandb
import comet_ml
import argparse
import yaml
# from torch import multiprocessing
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader
# try:
#     multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from VizDoom.VizDoom_src.train import train
from TMaze_new.TMaze_new_src.utils import set_seed, get_intro_maniskill
from ManiSkill.ManiSkill_src.utils import ManiSkillIterDataset

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker:.*")


# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  
# os.environ["OMP_NUM_THREADS"] = "1" 


# with open("wandb_config.yaml") as f:
#     wandb_config = yaml.load(f, Loader=yaml.FullLoader)
# os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

with open("ManiSkill/ManiSkill_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# python3 ManiSkill/ManiSkill_src/train_maniskill.py --model_mode 'RATE' --arch_mode 'TrXL' --ckpt_folder 'test' --text 'RATE' --nmt 5 --mem_len 165 --n_head_ca 2 --mrv_act 'relu' --skip_dec_ffn

def create_args():
    parser = argparse.ArgumentParser(description='RATE ManiSkill trainer') 

    parser.add_argument('--model_mode',     type=str, default='RATE',  help='Model training mode. Available variants: "DT, DTXL, RATE (Ours), RATEM (RMT)"')    
    parser.add_argument('--arch_mode',      type=str, default='TrXL',  help='Model architecture mode. Available variants: "TrXL", "TrXL-I", "GTrXL"')
    parser.add_argument('--start_seed',     type=int, default=1,       help='Start seed')
    parser.add_argument('--end_seed',       type=int, default=3,       help='End seed')
    parser.add_argument('--ckpt_folder',    type=str, default='ckpt',  help='Checkpoints directory')
    parser.add_argument('--text',           type=str, default='',      help='Short text description of rouns group')

    parser.add_argument('--nmt',            type=int, default=5,       help='')
    parser.add_argument('--mem_len',        type=int, default=165,     help='')
    parser.add_argument('--n_head_ca',      type=int, default=2,       help='')
    parser.add_argument('--mrv_act',        type=str, default='relu',  help='["no_act", "relu", "leaky_relu", "elu", "tanh"]')
    parser.add_argument('--skip_dec_ffn',   action='store_true',       help='Skip Feed Forward Network (FFN) in Decoder if set')
    parser.add_argument('--h5_to_npz',      action='store_true',       help='Convert .h5 files to .npz files if set. Do it only once at the beginning!')
    return parser

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")

    
    get_intro_maniskill()
    
    args = create_args().parse_args()
    model_mode = args.model_mode
    start_seed = args.start_seed
    end_seed = args.end_seed
    arch_mode = args.arch_mode
    ckpt_folder = args.ckpt_folder
    TEXT_DESCRIPTION = args.text
    mem_len = args.mem_len
    nmt = args.nmt
    n_head_ca = args.n_head_ca
    mrv_act = args.mrv_act
    skip_dec_ffn = args.skip_dec_ffn
    h5_to_npz = args.h5_to_npz

    config["model_mode"] = model_mode
    config["arctitecture_mode"] = arch_mode
    config["text_description"] = TEXT_DESCRIPTION
    config['model_config']['skip_dec_ffn'] = skip_dec_ffn

    for RUN in range(start_seed, end_seed+1):
        set_seed(RUN)
        print(f"Random seed set as {RUN}") 

        """ ARCHITECTURE MODE """
        if config["arctitecture_mode"] == "TrXL":
            config["model_config"]["use_gate"] = False
            config["model_config"]["use_stable_version"] = False
        elif config["arctitecture_mode"] == "TrXL-I":
            config["model_config"]["use_gate"] = False
            config["model_config"]["use_stable_version"] = True
        elif config["arctitecture_mode"] == "GTrXL":
            config["model_config"]["use_gate"] = True
            config["model_config"]["use_stable_version"] = True     

        print(f"Selected Architecture: {config['arctitecture_mode']}")  

        max_segments = config["training_config"]["sections"]

        """ MODEL MODE """
        if config["model_mode"] == "RATE": 
            config["model_config"]["mem_len"] = mem_len
            config["model_config"]["mem_at_end"] = True
            config["model_config"]["num_mem_tokens"] = nmt
            config["model_config"]["n_head_ca"] = n_head_ca
            config["model_config"]["mrv_act"] = mrv_act
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]
        

        elif config["model_mode"] == "DT":
            config["model_config"]["mem_len"] = 0
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["model_config"]["n_head_ca"] = 0
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]

        elif config["model_mode"] == "DTXL":
            config["model_config"]["mem_len"] = mem_len
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["model_config"]["n_head_ca"] = 0
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]

        elif config["model_mode"] == "RATEM":
            config["model_config"]["mem_len"] = 0
            config["model_config"]["mem_at_end"] = True
            config["model_config"]["num_mem_tokens"] = nmt
            config["model_config"]["n_head_ca"] = n_head_ca
            config["model_config"]["mrv_act"] = mrv_act
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]

        elif config["model_mode"] == "RATE_wo_nmt":
            print("Custom Mode!!! RATE wo nmt")
            config["model_config"]["mem_len"] = mem_len
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["model_config"]["n_head_ca"] = 0
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]
        
        if nmt == 0:
            config["model_config"]["mem_at_end"] = False
            
        print(f"Selected Model: {config['model_mode']}")  

        mini_text = f"arch_mode_{config['arctitecture_mode']}"
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace('-', '_')
        group = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}'
        name = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}_RUN_{RUN}_{date_time}'
        current_dir = os.getcwd()
        current_folder = os.path.basename(current_dir)
        ckpt_path = f'../{current_folder}/ManiSkill/ManiSkill_checkpoints/{ckpt_folder}/{name}/'
        isExist = os.path.exists(ckpt_path)
        if not isExist:
            os.makedirs(ckpt_path)

        experiment = None
        if config["wandb_config"]["wwandb"]:
            run = wandb.init(project=config['wandb_config']['project_name'], name=name, group=group, config=config, save_code=True, reinit=True)
        elif config['wandb_config']['wcomet']:
            print("*"*10, "COMET ML API SELECTED", "*"*10)

            comet_ml.init()

            # experiment =comet_ml.Experiment(
            #     api_key=wandb_config['comet_ml_api'],
            #     project_name=config['wandb_config']['project_name'],
            #     workspace=wandb_config['workspace_name'],
            #     log_code=True
            # )

            experiment.set_name(name)
            experiment.add_tags([group])
            experiment.log_parameters(config)
            

        #================================================== DATALOADERS CREATION ======================================================#
        
        # * IF USE ITER DATASET (5K TRAJECTORIES)
        path_to_splitted_dataset = 'ManiSkill/dataset_pushcube_v1/'
        train_dataset = ManiSkillIterDataset(path_to_splitted_dataset, 
                                            gamma=config["data_config"]["gamma"], 
                                            max_length=max_length, 
                                            normalize=config["data_config"]["normalize"],
                                            h5_to_npz=h5_to_npz)

        train_dataloader = DataLoader(train_dataset, 
                                     batch_size=config["training_config"]["batch_size"], 
                                     shuffle=True, 
                                     num_workers=1,
                                     persistent_workers=True,
                                     pin_memory=True) # * if process die again, try to set persistent_workers=True or num_workers=0


        print(f"Train: {len(train_dataloader) * config['training_config']['batch_size']} trajectories (first {max_length} steps)")

        if config["data_config"]["normalize"] == 0:
            type_norm = 'Without normalization'
        elif config["data_config"]["normalize"] == 1:
            type_norm = '/255.'
        elif config["data_config"]["normalize"] == 2:
            type_norm = 'Standard scaling'

        print(f'Normalization mode: {config["data_config"]["normalize"]}: {type_norm}')
        if config["data_config"]["normalize"] == 2:
            raise NotImplementedError("Standard scaling is not implemented for ManiSkill dataset")
        else:
            mean, std = None, None
        #==============================================================================================================================#
        wandb_step = 0
        model = train(ckpt_path, config, train_dataloader, mean, std, max_segments, experiment)
            
        torch.cuda.empty_cache()

        if config["wandb_config"]["wwandb"]:
            run.finish()
        elif config['wandb_config']['wcomet']:
            experiment.end()