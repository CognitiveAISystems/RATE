def add_env_specific_info_to_config(config: dict) -> dict:
    """Add environment-specific configuration parameters to the config dictionary.

    This function modifies the configuration dictionary based on the selected
    environment and model mode. It sets up specific parameters for different
    environments (TMaze, Memory Maze, POPGym, Minigrid Memory) and adjusts
    training and inference settings accordingly.

    Args:
        config: Configuration dictionary containing model and environment settings.
            Required keys:
            - model.env_name: Name of the environment (str)
            - model.model_mode: Model architecture mode (str)
            - max_n_final: Maximum number of final states (int, for TMaze)
            - training.context_length: Context length for training (int)
            - training.sections: Number of sections (int)
            - online_inference: Dictionary for inference settings

    Returns:
        dict: Updated configuration dictionary with environment-specific settings.

    Environment-specific modifications:
        TMaze:
            - Sets max_segments based on max_n_final
            - Configures episode timeout and corridor length based on model mode
            - Adjusts sections based on model architecture
            - Special handling for RATE, RMT, and TrXL models

        Memory Maze:
            - Enables only_non_zero_rewards filter

        POPGym and Minigrid Memory:
            - Sets max_segments equal to sections

    Notes:
        - For TMaze, different model modes (RATE, RMT, TrXL) have specific
          timeout and corridor length calculations
        - Memory Maze uses a reward filtering strategy
        - POPGym and Minigrid Memory use simpler segment configuration
    """
    if config["model"]["env_name"] == "tmaze":
        config["training"]["max_segments"] = config["max_n_final"]
        if config["model_mode"] in ["RATE", "RMT", "TrXL", "MATL"]:
            config["online_inference"]["episode_timeout"] = \
                config["max_n_final"] * config["training"]["context_length"]
            config["online_inference"]["corridor_length"] = \
                config["max_n_final"] * config["training"]["context_length"] - 2
        else:
            config["online_inference"]["episode_timeout"] = config["training"]["context_length"]
            config["online_inference"]["corridor_length"] = config["training"]["context_length"] - 2
        config["training"]["sections"] = config["max_n_final"]

        # Mltiple episode timeouts for final evaluation
        # config["online_inference"]["multiple_timeouts"] = [9, 30, 60, 90, 150, 210, 270, 360, 480, 600, 750, 900]

        if config["model_mode"] not in ["RATE", "RMT", "TrXL", "MATL"]:
            config["training"]["sections"] = 1
        else:
            config["training"]["sections"] = config["max_n_final"]
    elif config["model"]["env_name"] == "memory_maze":
        config["data"]["only_non_zero_rewards"] = True
    elif "popgym" in config["model"]["env_name"]:
        config["training"]["max_segments"] = config["training"]["sections"]
    elif "minigrid_memory" in config["model"]["env_name"]:
        config["training"]["max_segments"] = config["training"]["sections"]
    elif config["model"]["env_name"] in ['CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Acrobot-v1', 'Pendulum-v1']:
        # MDP environments configuration
        config["training"]["max_segments"] = config["training"]["sections"]

    return config

def configure_model_architecture(config: dict) -> tuple[int, int]:
    """Configure model architecture and mode settings.
    
    This function sets up the model architecture based on the selected mode
    (RATE, DT, DTXL, RMT, TrXL, etc.) and configures various model parameters
    including memory settings, attention mechanisms, and context lengths.
    It supports multiple architecture modes and model variants.

    Args:
        config: Configuration dictionary containing model settings.
            Required keys:
            - arctitecture_mode: Architecture type (str)
                Options: "TrXL", "TrXL-I", "GTrXL"
            - model_mode: Model variant (str)
                Options: "RATE", "DT", "DTXL", "RMT", "TrXL", "BC", "CQL", "IQL",
                        "DLSTM", "DMamba", "LSDT"
            - model: Dictionary containing model parameters
                - mem_len: Memory length (int)
                - mem_at_end: Memory position flag (bool)
                - num_mem_tokens: Number of memory tokens (int)
                - n_head_ca: Number of cross-attention heads (int)
                - mrv_act: Memory readout activation (str)
            - training: Dictionary containing training parameters
                - context_length: Base context length (int)
                - sections: Number of sections (int)

    Returns:
        tuple[int, int]: A tuple containing:
            - max_segments: Maximum number of segments for training
            - max_length: Maximum sequence length for the model

    Architecture Modes:
        TrXL: Basic Transformer XL without gates
        TrXL-I: Improved TrXL with stable version
        GTrXL: Gated Transformer XL with stable version

    Model Modes and their specific configurations:
        RATE:
            - Uses memory tokens at the end
            - Configures cross-attention heads
            - Sets memory readout activation
            - max_length = sections * context_length

        DT (Decision Transformer):
            - No memory tokens
            - Single section
            - max_length = context_length * sections

        DTXL:
            - Uses memory but no memory tokens
            - Single section
            - max_length = context_length * sections

        RMT (Recurrent Memory Transformer):
            - Memory tokens at the end
            - No cross-attention
            - max_length = sections * context_length

        TrXL:
            - Uses memory but no memory tokens
            - No cross-attention
            - max_length = sections * context_length

        BC/CQL/IQL/DLSTM/DMamba/LSDT:
            - Single section
            - max_length = context_length * sections

    Notes:
        - Memory tokens and memory position are interdependent
        - Context length and sections affect the maximum sequence length
        - Different model modes have specific memory and attention configurations
        - Architecture mode affects gate and stability settings
    """
    # Architecture mode configuration
    if config["arctitecture_mode"] == "TrXL":
        config["model"]["use_gate"] = False
        config["model"]["use_stable_version"] = False
    elif config["arctitecture_mode"] == "TrXL-I":
        config["model"]["use_gate"] = False
        config["model"]["use_stable_version"] = True
    elif config["arctitecture_mode"] == "GTrXL":
        config["model"]["use_gate"] = True
        config["model"]["use_stable_version"] = True     

    print(f"Selected Architecture: {config['arctitecture_mode']}")  

    max_segments = config["training"]["sections"]
    max_length = 0

    # Model mode configuration
    if config["model_mode"] == "RATE": 
        config["model"]["mem_len"] = config['model']['mem_len']
        config["model"]["mem_at_end"] = config["model"]["mem_at_end"]
        config["model"]["num_mem_tokens"] = config['model']['num_mem_tokens']
        config["model"]["n_head_ca"] = config['model']['n_head_ca']
        config["model"]["mrv_act"] = config['model']['mrv_act']
        max_length = config["training"]["sections"] * config["training"]["context_length"]

    elif config["model_mode"] == "DT":
        config["model"]["mem_len"] = 0
        config["model"]["mem_at_end"] = False
        config["model"]["num_mem_tokens"] = 0
        config["model"]["n_head_ca"] = 0
        config["training"]["context_length"] = config["training"]["context_length"] * config["training"]["sections"]
        config["training"]["sections"] = 1
        max_length = config["training"]["context_length"]

    elif config["model_mode"] == "DTXL":
        config["model"]["mem_len"] = config['model']['mem_len']
        config["model"]["mem_at_end"] = False
        config["model"]["num_mem_tokens"] = 0
        config["model"]["n_head_ca"] = 0
        config["training"]["context_length"] = config["training"]["context_length"] * config["training"]["sections"]
        config["training"]["sections"] = 1
        max_length = config["training"]["context_length"]

    elif config["model_mode"] == "RMT":
        config["model"]["mem_len"] = 0
        config["model"]["mem_at_end"] = True
        config["model"]["num_mem_tokens"] = config['model']['num_mem_tokens']
        config["model"]["n_head_ca"] = 0
        config["model"]["mrv_act"] = 'no_act'
        max_length = config["training"]["sections"] * config["training"]["context_length"]

    elif config["model_mode"] == "TrXL":
        config["model"]["mem_len"] = config['model']['mem_len']
        config["model"]["mem_at_end"] = False
        config["model"]["num_mem_tokens"] = 0
        config["model"]["n_head_ca"] = 0
        max_length = config["training"]["sections"] * config["training"]["context_length"]
    
    elif config["model_mode"] in ["BC", "CQL", "IQL", "DLSTM", "DMamba", "LSDT"]:
        config["training"]["context_length"] = config["training"]["context_length"] * config["training"]["sections"]
        config["training"]["sections"] = 1
        max_length = config["training"]["context_length"]
    
    elif config["model_mode"] == "MATL":
        config["model"]["memory_size"] = config["model"]["memory_size"]
        max_length = config["training"]["sections"] * config["training"]["context_length"]

    if config['model']['num_mem_tokens'] == 0:
        config["model"]["mem_at_end"] = False
        
    print(f"Selected Model: {config['model_mode']}")
    print('\n')
    
    return max_segments, max_length