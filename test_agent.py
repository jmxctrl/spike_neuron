import numpy as np
import sys
import os
from train import load_weights, list_saved_weights, run_episode

def test_trained_agent(weight_file, num_tests=5, max_steps=100):
    """
    Test a trained agent by running episodes without learning
    
    Params:
        weight_file: path to saved weight file
        num_tests: number of test episodes to run
        max_steps: maximum steps per episode
    
    Returns:
        test_rewards: list of rewards from test episodes
    """
    print("=" * 60)
    print("TESTING TRAINED AGENT")
    print("=" * 60)
    
    # Load the trained weights
    W = load_weights(weight_file)
    
    print(f"\nRunning {num_tests} test episodes (max {max_steps} steps each)...")
    print("-" * 60)
    
    test_rewards = []
    test_steps = []
    test_outcomes = []  # Track success/failure
    
    for test_num in range(num_tests):
        # Run episode without learning (just inference)
        spike_history, reward_history, total_reward, eligibility_history = run_episode(W, max_steps=max_steps)
        test_rewards.append(total_reward)
        
        num_steps = len(reward_history)
        test_steps.append(num_steps)
        
        # Determine outcome: success = completed all steps, failure = crashed early
        is_crashed = num_steps < max_steps
        test_outcomes.append(is_crashed)
        
        outcome_str = "❌ CRASHED" if is_crashed else "✅ SUCCESS"
        print(f"Test {test_num + 1}/{num_tests}: "
              f"Steps={num_steps}, Reward={total_reward:.2f} {outcome_str}")
    
    print("-" * 60)
    print("\n📊 TEST RESULTS:")
    print(f"  Average reward: {np.mean(test_rewards):.2f}")
    print(f"  Best reward: {max(test_rewards):.2f}")
    print(f"  Worst reward: {min(test_rewards):.2f}")
    print(f"  Std deviation: {np.std(test_rewards):.2f}")
    
    # Success Rate Analysis
    num_successes = sum(1 for outcome in test_outcomes if not outcome)
    success_rate = (num_successes / num_tests) * 100
    print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}% ({num_successes}/{num_tests})")
    
    # Failure Analysis
    failures = [i for i, outcome in enumerate(test_outcomes) if outcome]
    if failures:
        print(f"\n⚠️  FAILURE ANALYSIS:")
        print(f"  Failed episodes: {[f+1 for f in failures]}")
        failed_rewards = [test_rewards[i] for i in failures]
        failed_steps = [test_steps[i] for i in failures]
        print(f"  Failed rewards: {[f'{r:.2f}' for r in failed_rewards]}")
        print(f"  Failed steps: {failed_steps} (crashed before reaching {max_steps})")
        print(f"  Average steps before crash: {np.mean(failed_steps):.1f}")
    
    # Deployment Check
    print(f"\n🚀 DEPLOYMENT READINESS:")
    min_avg_reward = 200
    min_success_rate = 0.85
    
    avg_reward = np.mean(test_rewards)
    if avg_reward >= min_avg_reward and success_rate >= min_success_rate * 100:
        print(f"  ✅ READY FOR PRODUCTION")
    elif avg_reward >= min_avg_reward:
        print(f"  ⚠️  MARGINAL - High reward but low success rate ({success_rate:.1f}%)")
    elif success_rate >= min_success_rate * 100:
        print(f"  ⚠️  MARGINAL - Consistent but low reward ({avg_reward:.2f})")
    else:
        print(f"  ❌ NOT READY - Needs improvement")
    
    return test_rewards


if __name__ == "__main__":
    # List available weights
    list_saved_weights()
    
    # Check if weight file specified
    if len(sys.argv) > 1:
        weight_file = sys.argv[1]
    else:
        print("\n⚠️  No weight file specified!")
        print("Usage: python test_agent.py <path_to_weights.npy>")
        print("\nExample:")
        print("  python test_agent.py trained_weights/weights_reward294.8_20260301_143022.npy")
        sys.exit(1)
    
    # Test the agent
    test_rewards = test_trained_agent(weight_file, num_tests=10, max_steps=100)
    
    print("\n✓ Testing complete!")
