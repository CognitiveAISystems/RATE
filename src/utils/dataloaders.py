from torch.utils.data import random_split, DataLoader

def create_dataloader(config, max_length, segment_length):
    """
    Creates appropriate dataloader based on environment name
    
    Args:
        config: Configuration dictionary
        max_length: Maximum sequence length
        segment_length: Length of each segment
    
    Returns:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data (None for vizdoom)
    """
    if config["model"]["env_name"] == "vizdoom":
        from envs_datasets import ViZDoomIterDataset

        train_dataset = ViZDoomIterDataset(
            config["data"]["path_to_dataset"],
            gamma=config["data"]["gamma"],
            max_length=max_length,
            normalize=1
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        print(f"Train: {len(train_dataloader) * config['training']['batch_size']} trajectories (first {max_length} steps)")
        
        return train_dataloader
    
    elif config["model"]["env_name"] == "minigrid_memory":
        from envs_datasets import MinigridMemoryIterDataset

        train_dataset = MinigridMemoryIterDataset(
            config["data"]["path_to_dataset"], 
            gamma=config["data"]["gamma"], 
            max_length=max_length, 
            normalize=1
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["training"]["batch_size"], 
            shuffle=True, 
            num_workers=8,
            pin_memory=True
        )

        print(f"Train: {len(train_dataloader) * config['training']['batch_size']} trajectories (first {max_length} steps)")

        return train_dataloader

    elif config["model"]["env_name"] == "memory_maze":
        from envs_datasets import MemoryMazeDataset

        train_dataset = MemoryMazeDataset(
            config["data"]["path_to_dataset"],  
            gamma=config["data"]["gamma"], 
            max_length=max_length, 
            only_non_zero_rewards=config["data"]["only_non_zero_rewards"]
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["training"]["batch_size"], 
            shuffle=True, 
            num_workers=8,
            pin_memory=True
        )

        print(f"Train: {len(train_dataset)} trajectories ({max_length} steps / trajectory)")

        return train_dataloader
    
    elif "popgym" in config["model"]["env_name"]:
        from envs_datasets import POPGymDataset

        train_dataset = POPGymDataset(
            config["data"]["path_to_dataset"],  
            gamma=config["data"]["gamma"], 
            max_length=max_length,
            env_name=config["model"]["env_name"]
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["training"]["batch_size"], 
            shuffle=True, 
            num_workers=8,
            pin_memory=True
        )

        print(f"Train: {len(train_dataset)} trajectories ({max_length} steps / trajectory)")

        return train_dataloader
    
    elif "mikasa_robo" in config["model"]["env_name"]:
        from envs_datasets import MIKASARoboIterDataset

        train_dataset = MIKASARoboIterDataset(
            config["data"]["path_to_dataset"],
            gamma=config["data"]["gamma"],
            max_length=max_length,
            normalize=1
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        print(f"Train: {len(train_dataloader) * config['training']['batch_size']} trajectories (first {max_length} steps)")
        
        return train_dataloader

    elif config["model"]["env_name"] == "tmaze":
        from envs_datasets import TMaze_data_generator, TMazeCombinedDataLoader
        
        TMaze_data_generator(
            max_segments=config["training"]["max_segments"],
            multiplier=1000,
            hint_steps=1, 
            desired_reward=1,
            win_only=True,
            segment_length=segment_length
        )

        combined_dataloader = TMazeCombinedDataLoader(
            n_init=config["min_n_final"],
            n_final=config["max_n_final"],
            multiplier=1000,
            hint_steps=1,
            batch_size=config["training"]["batch_size"],
            mode="",
            cut_dataset=False,
            one_mixed_dataset=True,
            desired_reward=1,
            win_only=True,
            segment_length=segment_length
        )
        
        full_dataset = combined_dataloader.dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        print(f'Number of considered segments: {config["max_n_final"]}, dataset length: {len(combined_dataloader.dataset)}, Train: {len(train_dataset)}')
        return train_dataloader
    
    else:
        raise ValueError(f"Unknown environment: {config['model']['env_name']}")