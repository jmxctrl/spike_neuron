import numpy as np 
from light_task import CarState, get_sensor_values, encode_sensors_to_spikes
from spike_vectorized import run_vectorized_lif, create_weight_matrix
from neuromodulation import classify_actions, reward_calculator, compute_stdp_eligibility, apply_dopamine


# ============================================
# Episode-based Training for SNN Car Control
# ============================================

""" 
Run the car simulator
Let the SNN control it
Collect data about what happened
Update weights based on success/failure
Repeat until it learns to drive

run_episode(): collect data 
train(): learning and orchestrating 

"""


def train_snn(num_episodes=100, learning_rate=0.01, baseline_lr=0.1):

    """
    the function teaches SNN to drive the car by running many episodes to learn based 
    reward 

    params:     
        weights 
        episode parameters 


    return: weights, episode reward, baseline history
    """

    # initialize 1) weight matrix 2) baseline 3) track progress 
    W = create_weight_matrix(100)
    baseline = 0.0 # start value at ep 0
    episode_rewards = []
    
    # Track best weights
    best_reward = -np.inf
    best_weights = W.copy()

    # loop through episodes 
    for episode in range(num_episodes):
        # 1. run episode 
        spike_history, reward_history, total_reward = run_episode(W)

        # 2. compute eligibility 
        eligibility = compute_stdp_eligibility(spike_history)  # Shape: (N, N)
        
        # Expand eligibility to match timesteps (same eligibility for all timesteps)
        T = len(reward_history)
        eligibility_history = np.array([eligibility] * T)  # Shape: (T, N, N)

        # 3. update weights AND baseline 
        W, baseline = apply_dopamine(
            W, 
            eligibility_history, 
            reward_history, 
            baseline, 
            baseline_lr=baseline_lr, 
            learning_rate=learning_rate
        )

        # 4. append to rewards 
        episode_rewards.append(total_reward)
        
        # 5. track best weights
        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = W.copy()
            print(f" New best! Episode {episode}: reward={total_reward:.2f}")
        
        # 6. print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}: reward={total_reward:.2f}, baseline={baseline:.2f}")
    
    print(f"\n Returning BEST weights (from episode with reward={best_reward:.2f})")
    return best_weights, episode_rewards

def run_episode(weights, max_steps=100):
    """
    the function collects data about the car with current weights 

    params:
        weights
        max steps 

    return: spike_history, reward_history, total_reward 

    """
    car = CarState(position=0.0, speed=0.1) # start centered 
    
    spike_history = []
    reward_history = []

    total_reward = 0.0
    prev_action = 0 # start with straight 

    for t in range(max_steps):
        # get sensors from the car 
        sensors = get_sensor_values(car)

        # encode sensors to spikes 
        input_spikes = encode_sensors_to_spikes(sensors, T=10, p_max=0.2)

        # run SNN forward pass 
        spikes, voltages, pathway_history = run_vectorized_lif(
            W=weights,
            external_spikes=input_spikes,
            num_steps=10,
            num_neurons=100,
            plot=False
        )

        # get action from SNN output 
        action = classify_actions(pathway_history)

        # update car position -> car.update(action)
        car.update(action)

        # calculate reward 
        reward = reward_calculator(car, action, prev_action)

        # store data 
        spike_history.append(spikes[-1, :]) #last row
        reward_history.append(reward)
        total_reward += reward 
        prev_action = action 

        # check if crashed 
        if car.is_crashed():
            break 
    
    # Convert lists to numpy arrays
    spike_history = np.array(spike_history)  # Shape: (T, 100)
    reward_history = np.array(reward_history)  # Shape: (T,)
    
    return spike_history, reward_history, total_reward


# ============================================
# Test/Demo Section
# ============================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("TRAINING SNN TO DRIVE CAR")
    print("=" * 60)
    
    # Train the agent
    W_trained, episode_rewards = train_snn(num_episodes=500, learning_rate=0.1, baseline_lr=0.1)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final episode reward: {episode_rewards[-1]:.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, marker='o', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('SNN Training Progress: Learning to Drive', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Training visualization complete!")
