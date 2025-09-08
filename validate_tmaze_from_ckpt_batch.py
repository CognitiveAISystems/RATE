import torch
import json
from offline_rl_baselines.MATL import MATLModel

from src.validation.val_tmaze import get_returns_TMaze
from src.utils.set_seed import set_seed, seeds_list

import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import concurrent.futures
from src.envs.tmaze.tmaze import TMazeClassicPassive


def parse_args():
    parser = argparse.ArgumentParser(description='Validate models on TMaze')
    parser.add_argument('--dir_to_ckpts', type=str, default='runs/NeurIPS_2025/MATL/T_9', 
                        help='Path to the directory with the checkpoints')
    parser.add_argument('--list_of_best_ckpts', type=int, nargs='+', default=[5, 5, 5, 5, 5], 
                        help='List of best ckpts for each model')
    return parser.parse_args()


# LIST_OF_EPISODE_LENGTHS = [9, 30, 90, 150, 300, 600, 900, 1200, 2400, 4800, 9600]
LIST_OF_EPISODE_LENGTHS = [1000000]


def validate_model(model_config, ckpt_path, model_name, device):
    # Инициализация и загрузка модели
    matl_config = model_config["model"].copy()
    matl_config["dtype"] = model_config["dtype"]
    # Set default sequence_format if not specified
    if matl_config.get("sequence_format") is None:
        matl_config["sequence_format"] = "sra"
    model = MATLModel(**matl_config)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    batch_size = len(seeds_list)
    results = {}
    
    for T in LIST_OF_EPISODE_LENGTHS:
        rewards, successes = get_returns_TMaze(
            model=model,
            ret=1.0,
            seeds=seeds_list,
            episode_timeout=T,
            corridor_length=T-2,
            context_length=model_config["training"]["context_length"],
            device=device,
            config=model_config,
            create_video=False
        )
        
        success_rate = successes/batch_size
        mean_reward = sum(rewards)/batch_size
        
        print(f"\n{model_name} - T={T}:")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Mean reward: {mean_reward:.3f}")
        
        results[T] = success_rate
    
    return results


def main(args):
    dir_to_ckpts = args.dir_to_ckpts
    list_of_best_ckpts = args.list_of_best_ckpts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)  # Для воспроизводимости
    
    # Ищем директории, содержащие RUN_* в названии
    model_dirs = []
    for root, dirs, files in os.walk(dir_to_ckpts):
        for dir_name in dirs:
            if "RUN_" in dir_name:
                run_num = int(dir_name.split("RUN_")[1].split("_")[0])
                model_dirs.append((run_num, os.path.join(root, dir_name)))
    
    # Сортируем по номеру RUN
    model_dirs.sort()
    
    # Проверяем, что нашли нужное количество директорий
    if len(model_dirs) < len(list_of_best_ckpts):
        raise ValueError(f"Found only {len(model_dirs)} model directories, but need {len(list_of_best_ckpts)}")
    
    # Берем первые N директорий, где N = длина списка list_of_best_ckpts
    model_dirs = model_dirs[:len(list_of_best_ckpts)]
    
    print(f"Found {len(model_dirs)} model directories:")
    for run_num, dir_path in model_dirs:
        print(f"RUN_{run_num}: {dir_path}")
    
    # Формируем пути к чекпоинтам и конфигам
    checkpoint_paths = []
    config_paths = []
    model_configs = []
    
    for i, (run_num, model_dir) in enumerate(model_dirs):
        ckpt_path = f"{model_dir}/checkpoints/step_{list_of_best_ckpts[i]}.pth"
        config_path = f"{model_dir}/config.json"
        
        # Проверяем существование файлов
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for RUN_{run_num} not found: {ckpt_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config for RUN_{run_num} not found: {config_path}")
        
        checkpoint_paths.append(ckpt_path)
        config_paths.append(config_path)
        
        # Загружаем конфиг
        with open(config_path, "r") as f:
            model_configs.append(json.load(f))
    
    # Создаем список моделей для валидации
    models = []
    for i, (run_num, _) in enumerate(model_dirs):
        models.append({
            "config": model_configs[i], 
            "ckpt_path": checkpoint_paths[i], 
            "name": f"Model {run_num}"
        })
    
    # Параллельный запуск валидации моделей
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for model_info in models:
            future = executor.submit(
                validate_model, 
                model_info["config"], 
                model_info["ckpt_path"], 
                model_info["name"],
                device
            )
            futures.append(future)
        
        # Получаем результаты
        all_results = {}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            model_name = models[i]["name"]
            all_results[model_name] = future.result()

    results_df = pd.DataFrame(all_results)
    mean_sr = results_df.mean(axis=1)
    std_sr = results_df.std(axis=1)
    sem_sr = std_sr / np.sqrt(results_df.shape[1]-1)
    results_df.insert(0, 'T', results_df.index)
    results_df['mean'] = mean_sr
    results_df['std'] = std_sr
    results_df['sem'] = sem_sr
    
    # Создание директории для результатов на основе переданного пути
    result_dir = f'tmaze_val_results/{"/".join(dir_to_ckpts.split("/")[-2:])}'
    os.makedirs(result_dir, exist_ok=True)
    results_df.to_csv(f'{result_dir}/results_df.csv', index=False)
    
    # Здесь можно добавить сравнение результатов всех моделей или вывод итоговой статистики
    print("\n=== VALIDATION COMPLETE FOR ALL MODELS ===")


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python validate_tmaze_from_ckpt_batch.py --dir_to_ckpts runs/NeurIPS_2025/MATL/T_9 --list_of_best_ckpts 5 5 5 5 5