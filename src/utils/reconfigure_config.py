def add_env_specific_info_to_config(config):
    if config["model"]["env_name"] == "tmaze":
        config["training"]["max_segments"] = config["max_n_final"]
        if config["model_mode"] in ["RATE", "RMT", "TrXL"]:
            config["online_inference"]["episode_timeout"] = \
                config["max_n_final"] * config["training"]["context_length"]
            config["online_inference"]["corridor_length"] = \
                config["max_n_final"] * config["training"]["context_length"] - 2
        else:
            config["online_inference"]["episode_timeout"] = config["training"]["context_length"]
            config["online_inference"]["corridor_length"] = config["training"]["context_length"] - 2
        config["training"]["sections"] = config["max_n_final"]

        # Mltiple episode timeouts for final evaluation
        config["online_inference"]["multiple_timeouts"] = [9, 30, 60, 90, 150, 210, 270, 360, 480, 600, 750, 900]

        if config["model_mode"] not in ["RATE", "RMT", "TrXL"]:
            config["training"]["sections"] = 1
        else:
            config["training"]["sections"] = config["max_n_final"]
    elif config["model"]["env_name"] == "memory_maze":
        config["data"]["only_non_zero_rewards"] = True
    elif "popgym" in config["model"]["env_name"]:
        config["training"]["max_segments"] = config["training"]["sections"]

    return config

def configure_model_architecture(config: dict) -> tuple[int, int]:
    """Configure model architecture and mode settings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple[int, int]: max_segments and max_length
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
    
    elif config["model_mode"] in ["BC", "CQL", "IQL"]:
        config["training"]["context_length"] = config["training"]["context_length"] * config["training"]["sections"]
        config["training"]["sections"] = 1
        max_length = config["training"]["context_length"]

    elif config['model_mode'] == 'TT':  # TrajectoryTransformer
        config["model"]["mem_len"] = 0
        config["model"]["mem_at_end"] = False
        config["model"]["num_mem_tokens"] = 0
        config["model"]["n_head_ca"] = 0
        config["training"]["context_length"] = config["training"]["context_length"] * config["training"]["sections"]
        config["training"]["sections"] = 1
        max_length = config["training"]["context_length"]
    
    if config['model']['num_mem_tokens'] == 0:
        config["model"]["mem_at_end"] = False
        
    print(f"Selected Model: {config['model_mode']}")
    print('\n')
    
    return max_segments, max_length