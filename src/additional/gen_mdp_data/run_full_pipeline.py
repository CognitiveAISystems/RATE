"""
Full MDP Data Generation Pipeline

This script runs the complete pipeline for MDP data generation:
1. Train online RL policies on MDP environments
2. Collect offline datasets from trained policies  
3. Analyze the collected datasets

Usage:
    python run_full_pipeline.py --env all
    python run_full_pipeline.py --env CartPole-v1 --skip-training
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List
import json
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


class MDPPipeline:
    """Main pipeline class for MDP data generation."""
    
    ENVIRONMENTS = [
        "CartPole-v1",
        "MountainCar-v0", 
        "MountainCarContinuous-v0",
        "Acrobot-v1",
        "Pendulum-v1"
    ]
    
    def __init__(self, args):
        self.args = args
        self.script_dir = Path(__file__).parent
        self.results = {
            'training': {},
            'collection': {},
            'analysis': {}
        }
        
        # Determine environments to process
        if args.env == "all":
            self.environments = self.ENVIRONMENTS
        else:
            self.environments = [args.env]
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and handle errors."""
        print(f"\n{'='*60}")
        print(f"RUNNING: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=self.args.timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                if result.stdout:
                    print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
                return True
            else:
                print(f"❌ {description} failed with return code {result.returncode}")
                if result.stderr:
                    print("STDERR:", result.stderr)
                if result.stdout:
                    print("STDOUT:", result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out after {self.args.timeout} seconds")
            return False
        except Exception as e:
            print(f"💥 {description} failed with exception: {e}")
            return False
    
    def step_1_train_policies(self) -> bool:
        """Step 1: Train online RL policies."""
        if self.args.skip_training:
            print("\n🚫 Skipping training step (--skip-training flag)")
            return True
            
        print(f"\n🎯 STEP 1: Training RL policies")
        print(f"Environments: {', '.join(self.environments)}")
        
        success = True
        for env_name in self.environments:
            cmd = [
                "python", "train_online_rl.py",
                "--env", env_name,
                "--algorithm", "best",
                "--seed", str(self.args.seed),
                "--n-envs", str(self.args.n_envs),
                "--device", self.args.device
            ]
            
            if not self.run_command(cmd, f"Training {env_name}"):
                success = False
                self.results['training'][env_name] = 'FAILED'
            else:
                self.results['training'][env_name] = 'SUCCESS'
        
        return success
    
    def step_2_collect_datasets(self) -> bool:
        """Step 2: Collect offline datasets."""
        print(f"\n📊 STEP 2: Collecting datasets")
        
        success = True
        for env_name in self.environments:
            cmd = [
                "python", "collect_mdp_datasets.py",
                "--env", env_name,
                "--n-episodes", str(self.args.n_episodes),
                "--max-episode-length", str(self.args.max_episode_length),
                "--seed", str(self.args.seed)
            ]
            
            if self.args.model_path:
                cmd.extend(["--model-path", self.args.model_path])
            if self.args.algorithm:
                cmd.extend(["--algorithm", self.args.algorithm])
            
            if not self.run_command(cmd, f"Collecting data for {env_name}"):
                success = False
                self.results['collection'][env_name] = 'FAILED'
            else:
                self.results['collection'][env_name] = 'SUCCESS'
        
        return success
    
    def step_3_analyze_datasets(self) -> bool:
        """Step 3: Analyze collected datasets."""
        print(f"\n📈 STEP 3: Analyzing datasets")
        
        cmd = [
            "python", "analyze_datasets.py",
            "--env", "all" if len(self.environments) > 1 else self.environments[0],
            "--data-base-dir", self.args.data_base_dir,
            "--output-dir", self.args.output_dir
        ]
        
        if self.args.plot:
            cmd.append("--plot")
        
        success = self.run_command(cmd, "Analyzing datasets")
        self.results['analysis']['status'] = 'SUCCESS' if success else 'FAILED'
        
        return success
    
    def save_pipeline_results(self):
        """Save pipeline execution results."""
        results_file = self.script_dir / "pipeline_results.json"
        
        pipeline_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'arguments': vars(self.args),
            'environments': self.environments,
            'results': self.results,
            'success_rate': self._calculate_success_rate()
        }
        
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"\n📋 Pipeline results saved to: {results_file}")
    
    def _calculate_success_rate(self) -> dict:
        """Calculate success rates for each step."""
        success_rates = {}
        
        for step, step_results in self.results.items():
            if step_results:
                total = len(step_results)
                successful = sum(1 for status in step_results.values() if status == 'SUCCESS')
                success_rates[step] = f"{successful}/{total} ({100*successful/total:.1f}%)"
            else:
                success_rates[step] = "0/0 (N/A)"
        
        return success_rates
    
    def print_summary(self):
        """Print pipeline execution summary."""
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        print(f"Environments processed: {', '.join(self.environments)}")
        print(f"Total execution time: {time.time() - self.start_time:.1f} seconds")
        
        success_rates = self._calculate_success_rate()
        for step, rate in success_rates.items():
            print(f"{step.title()} success rate: {rate}")
        
        # Print detailed results
        for step, step_results in self.results.items():
            if step_results:
                print(f"\n{step.title()} Results:")
                for env_name, status in step_results.items():
                    status_icon = "✅" if status == "SUCCESS" else "❌"
                    print(f"  {status_icon} {env_name}: {status}")
    
    def run(self):
        """Run the complete pipeline."""
        self.start_time = time.time()
        
        print(f"\n🚀 Starting MDP Data Generation Pipeline")
        print(f"Environments: {', '.join(self.environments)}")
        print(f"Arguments: {vars(self.args)}")
        
        steps = [
            ("Training RL Policies", self.step_1_train_policies),
            ("Collecting Datasets", self.step_2_collect_datasets), 
            ("Analyzing Datasets", self.step_3_analyze_datasets)
        ]
        
        overall_success = True
        
        for step_name, step_func in steps:
            print(f"\n🔄 Starting: {step_name}")
            
            if not step_func():
                print(f"⚠️  {step_name} had failures")
                overall_success = False
                
                if self.args.stop_on_error:
                    print("🛑 Stopping pipeline due to --stop-on-error flag")
                    break
            else:
                print(f"✅ {step_name} completed successfully")
        
        # Save results and print summary
        self.save_pipeline_results()
        self.print_summary()
        
        if overall_success:
            print(f"\n🎉 Pipeline completed successfully!")
        else:
            print(f"\n⚠️  Pipeline completed with some failures")
            
        return overall_success


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete MDP data generation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment selection
    parser.add_argument("--env", type=str, default="all",
                       choices=["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                               "Acrobot-v1", "Pendulum-v1", "all"],
                       help="Environment(s) to process")
    
    # Training parameters
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip the training step (use existing models)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments for training")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device for training")
    
    # Collection parameters
    parser.add_argument("--n-episodes", type=int, default=1000,
                       help="Number of episodes to collect per environment")
    parser.add_argument("--max-episode-length", type=int, default=1000,
                       help="Maximum episode length")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Specific model path (overrides auto-detection)")
    parser.add_argument("--algorithm", type=str, default=None,
                       help="Algorithm name (required if using --model-path)")
    
    # Analysis parameters
    parser.add_argument("--data-base-dir", type=str, default="data/MDP",
                       help="Base directory for dataset storage")
    parser.add_argument("--output-dir", type=str, default="analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    
    # Pipeline control
    parser.add_argument("--stop-on-error", action="store_true",
                       help="Stop pipeline execution on first error")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout in seconds for each step")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_path and not args.algorithm:
        parser.error("--algorithm is required when using --model-path")
    
    # Run pipeline
    pipeline = MDPPipeline(args)
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
