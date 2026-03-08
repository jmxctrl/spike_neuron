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
    
    for test_num in range(num_tests):
        # Run episode without learning (just inference)
        spike_history, reward_history, total_reward, eligibility_history = run_episode(W, max_steps=max_steps)
        test_rewards.append(total_reward)
        
        num_steps = len(reward_history)
        print(f"Test {test_num + 1}/{num_tests}: "
              f"Steps={num_steps}, Reward={total_reward:.2f}")
    
    print("-" * 60)
    print("\n📊 TEST RESULTS:")
    print(f"  Average reward: {np.mean(test_rewards):.2f}")
    print(f"  Best reward: {max(test_rewards):.2f}")
    print(f"  Worst reward: {min(test_rewards):.2f}")
    print(f"  Std deviation: {np.std(test_rewards):.2f}")
    
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
