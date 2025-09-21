# Test setup
python src/additional/gen_mdp_data/test_mdp_setup.py

# Run complete pipeline for all environments
python src/additional/gen_mdp_data/run_full_pipeline.py --env all

# Run for specific environment
python src/additional/gen_mdp_data/run_full_pipeline.py --env CartPole-v1

# Individual steps
python src/additional/gen_mdp_data/train_online_rl.py --env CartPole-v1 --algorithm PPO
python src/additional/gen_mdp_data/collect_mdp_datasets.py --env CartPole-v1 --n-episodes 1000
python src/additional/gen_mdp_data/analyze_datasets.py --env all --plot


python src/additional/gen_mdp_data/train_online_rl.py --env CartPole-v1 --algorithm PPO
python src/additional/gen_mdp_data/collect_mdp_datasets.py --env CartPole-v1 --n-episodes 3000
python src/additional/gen_mdp_data/train_online_rl.py --env Pendulum-v1 --algorithm SAC
python src/additional/gen_mdp_data/collect_mdp_datasets.py --env Pendulum-v1 --n-episodes 5000