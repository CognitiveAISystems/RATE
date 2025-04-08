import os
import pandas as pd
from generate_popgym_train_scripts import generate_popgym_command, get_env_index

def main():
    csv_path = "run_experiments/POPGym/popgym_envs_info.csv"
    
    output_dir = "run_experiments/POPGym/run_scripts"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path, sep=';')
    
    for index, row in df.iterrows():
        env_full_name = row['Environment']
        
        env_name = env_full_name.replace('popgym-', '').replace('-v0', '')
        
        try:
            env_index = get_env_index(env_name)
            command = generate_popgym_command(env_name, env_index)
            
            output_path = os.path.join(output_dir, f"run_{env_index}.sh")
            
            with open(output_path, 'w') as f:
                f.write(command)
            
            os.chmod(output_path, 0o755)
            
            print(f"Script created for environment {env_full_name}: {output_path}")
            
        except Exception as e:
            print(f"Error creating script for environment {env_full_name}: {e}")
    
    print(f"Total scripts created: {len(df)}")

if __name__ == "__main__":
    main()