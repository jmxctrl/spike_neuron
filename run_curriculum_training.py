import numpy as np
import matplotlib.pyplot as plt
from train import train_snn_curriculum, save_weights

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING SNN WITH CURRICULUM LEARNING")
    print("=" * 60)
    
    # Train the agent with curriculum
    W_trained, episode_rewards = train_snn_curriculum(
        num_episodes=1000, 
        learning_rate=0.05,  # Try 0.05 or 0.1
        baseline_lr=0.1
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    best_reward = max(episode_rewards)
    print(f"Final episode reward: {episode_rewards[-1]:.2f}")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    
    # Save the trained weights
    save_weights(W_trained, reward=best_reward)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, marker='o', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('SNN Curriculum Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Training visualization complete!")