def print_config(config, indent_level=0):
    # from pprint import pprint
    # pprint(self.config, indent=2, width=80)
    COLORS = {
        'wandb': '\033[1;91m',            # Red
        'training': '\033[1;92m',         # Green
        'model': '\033[1;93m',            # Yellow
        'online_inference': '\033[1;95m', # Magenta
        'data': '\033[1;94m',             # Cyan
        'RESET': '\033[1;0m'              # Reset
    }
    for key, value in config.items():
        color = COLORS.get(key, COLORS['RESET'])
        indent = "    " * indent_level
        
        if isinstance(value, dict):
            print(f"{indent}{color}{key}:{COLORS['RESET']}")
            print_config(value, indent_level + 1)
        else:
            print(f"{indent}{key}: {value}")