from torch.utils.data import random_split, DataLoader


def create_dataloader(config: dict, max_length: int, segment_length: int) -> DataLoader:
    """Creates and configures appropriate DataLoader based on the environment name.

    This function serves as a factory for creating DataLoaders for different environments.
    It supports multiple environments including ViZDoom-Two-Colors, Minigrid Memory, Memory Maze,
    POPGym, MIKASA-Robo, and TMaze. Each environment has its specific dataset class
    and configuration requirements.

    Args:
        config: Configuration dictionary containing environment and training settings.
            Required keys:
            - model.env_name: Name of the environment (str)
            - data.path_to_dataset: Path to dataset directory (str)
            - data.gamma: Discount factor for reward calculation (float)
            - data.only_non_zero_rewards: Filter for non-zero rewards (bool, optional)
            - training.batch_size: Batch size for training (int)
            - training.max_segments: Maximum number of segments (int, for TMaze)
            - min_n_final: Minimum number of final states (int, for TMaze)
            - max_n_final: Maximum number of final states (int, for TMaze)
        max_length: Maximum sequence length for trajectories.
            Determines how many steps from each trajectory are used.
        segment_length: Length of each trajectory segment.
            Used for environments that require trajectory segmentation.

    Returns:
        DataLoader: Configured DataLoader for the specified environment.
            For most environments, returns only training DataLoader.
            For TMaze, returns training DataLoader with validation split.

    Supported Environments:
        - "vizdoom": ViZDoom environment with RGB observations
        - "minigrid_memory": Minigrid environment with memory requirements
        - "memory_maze": Memory Maze environment
        - "popgym": POPGym environments (Battleship, Minesweeper, etc.)
        - "mikasa_robo": MIKASA-Robo manipulation environment
        - "tmaze": TMaze environment with hint-based navigation

    Notes:
        - Each environment uses its specific dataset class with appropriate
          preprocessing and data loading strategies.
        - For ViZDoom and Minigrid Memory, observations are normalized.
        - TMaze environment includes validation split (80% train, 20% validation).
        - Batch sizes and number of workers are optimized per environment.
        - All DataLoaders use pin_memory=True for faster data transfer to GPU.
        - For TMaze, additional parameters control hint steps and reward conditions.

    Examples:
        >>> config = {
        ...     "model": {"env_name": "vizdoom"},
        ...     "data": {
        ...         "path_to_dataset": "path/to/data",
        ...         "gamma": 0.99
        ...     },
        ...     "training": {"batch_size": 32}
        ... }
        >>> dataloader = create_dataloader(config, max_length=150, segment_length=50)
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
            persistent_workers=True,
            num_workers=2, # 4
            pin_memory=True
        )
        
        print(f"Train: {len(train_dataloader) * config['training']['batch_size']} trajectories (first {max_length} steps)")
        
        return train_dataloader

    elif config["model"]["env_name"] == "tmaze":
        from envs_datasets import TMaze_data_generator, TMazeCombinedDataLoader
        TMaze_data_generator(
            max_segments=config["training"]["max_segments"],
            multiplier=1000, # 1000
            hint_steps=1, 
            desired_reward=1,
            win_only=True,
            segment_length=segment_length
        )

        combined_dataloader = TMazeCombinedDataLoader(
            n_init=config["min_n_final"],
            n_final=config["max_n_final"],
            multiplier=1000, # 1000
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