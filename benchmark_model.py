import numpy as np
import sys
from test_agent import test_trained_agent

def benchmark_model(weight_file, num_runs=5, tests_per_run=10):
    """
    Run multiple test runs to get statistical confidence
    
    Params:
        weight_file: path to weights
        num_runs: how many test runs (default 5)
        tests_per_run: episodes per run (default 10)
    """
    print("=" * 70)
    print(f"BENCHMARKING MODEL: {weight_file}")
    print(f"Running {num_runs} test runs with {tests_per_run} episodes each")
    print("=" * 70)
    
    all_rewards = []
    all_success_rates = []
    
    for run_num in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_num + 1}/{num_runs}")
        print(f"{'='*70}")
        
        test_rewards = test_trained_agent(weight_file, num_tests=tests_per_run, max_steps=100)
        all_rewards.extend(test_rewards)
        
        # Calculate success rate for this run
        success_rate = sum(1 for r in test_rewards if r > 100) / len(test_rewards)  # Crash = low reward
        all_success_rates.append(success_rate)
    
    # Final aggregated statistics
    print(f"\n{'='*70}")
    print("📊 AGGREGATED BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Total episodes: {len(all_rewards)}")
    print(f"Average reward (all runs): {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Best reward: {np.max(all_rewards):.2f}")
    print(f"Worst reward: {np.min(all_rewards):.2f}")
    print(f"Median reward: {np.median(all_rewards):.2f}")
    
    print(f"\nAverage success rate: {np.mean(all_success_rates)*100:.1f}% ± {np.std(all_success_rates)*100:.1f}%")
    print(f"Best run: {np.max(all_success_rates)*100:.1f}%")
    print(f"Worst run: {np.min(all_success_rates)*100:.1f}%")
    
    # Deployment decision
    print(f"\n{'='*70}")
    print("🚀 PRODUCTION DEPLOYMENT DECISION")
    print(f"{'='*70}")
    
    avg_reward = np.mean(all_rewards)
    avg_success = np.mean(all_success_rates)
    
    if avg_reward >= 220 and avg_success >= 0.85:
        print("✅ APPROVED FOR PRODUCTION")
        print(f"   Meets both criteria: reward={avg_reward:.2f}, success={avg_success*100:.1f}%")
    else:
        print("❌ NOT APPROVED - Needs improvement")
        if avg_reward < 220:
            print(f"   - Reward too low: {avg_reward:.2f} < 220")
        if avg_success < 0.85:
            print(f"   - Success rate too low: {avg_success*100:.1f}% < 85%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        weight_file = sys.argv[1]
    else:
        print("Usage: python benchmark_model.py <path_to_weights.npy>")
        sys.exit(1)
    
    benchmark_model(weight_file, num_runs=5, tests_per_run=10)
