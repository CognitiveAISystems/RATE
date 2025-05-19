import subprocess
from multiprocessing import Pool
import argparse

def run_collect_batch_of_data(args):
    batch_idx, random = args
    subprocess.run(['python', 'src/additional/gen_minigrid_memory_data/collect_batch_of_data.py', 
                   '--batch', str(batch_idx),
                   '--random', str(random)])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', type=str, choices=['True', 'False'], required=True,
                       help='Whether to use random environment (True/False)')
    args = parser.parse_args()

    process_args = [(i, args.random) for i in range(10)]

    # Create a pool of 10 processes
    with Pool(processes=10) as pool:
        pool.map(run_collect_batch_of_data, process_args)

"""
python gen_minigrid_memory_data.py --random True
# or
python gen_minigrid_memory_data.py --random False
"""