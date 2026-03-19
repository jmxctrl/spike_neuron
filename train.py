import numpy as np 
import os
from datetime import datetime
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

# ============================================
# Save/Load Functions
# ============================================

def save_weights(weights, filename=None, reward=None):
    """
    Save trained weights to disk
    
    Params:
        weights: (N, N) weight matrix to save
        filename: optional custom filename. If None, generates timestamp-based name
        reward: optional best reward to include in filename
    
    Returns:
        filepath: path where weights were saved
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if reward is not None:
            filename = f"weights_reward{reward:.1f}_{timestamp}.npy"
        else:
            filename = f"weights_{timestamp}.npy"
    
    # Create weights directory if it doesn't exist
    weights_dir = "trained_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    filepath = os.path.join(weights_dir, filename)
    np.save(filepath, weights)
    
    print(f"Weights saved to: {filepath}")
    return filepath


def load_weights(filepath):
    """
    Load trained weights from disk
    
    Params:
        filepath: path to .npy file containing weights
    
    Returns:
        weights: (N, N) weight matrix
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weight file not found: {filepath}")
    
    weights = np.load(filepath)
    print(f"Weights loaded from: {filepath}")
    print(f"   Shape: {weights.shape}")
    
    return weights


def list_saved_weights(weights_dir="trained_weights"):
    """
    List all saved weight files
    
    Returns:
        list of weight filenames
    """
    if not os.path.exists(weights_dir):
        print(f"No weights directory found at: {weights_dir}")
        return []
    
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.npy')]
    
    if weight_files:
        print(f"\n Found {len(weight_files)} saved weight file(s):")
        for i, f in enumerate(sorted(weight_files), 1):
            filepath = os.path.join(weights_dir, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {i}. {f} ({size_kb:.1f} KB)")
    else:
        print("No saved weights found.")
    
    return weight_files

def lr_scheduler(current_episode, total_episodes, warmup_episodes, max_lr, min_lr):
    """
    schedules a warm up period and cosine decay for learning rate during training 

    returns learning rate at time t 
    """

    if current_episode < warmup_episodes:
        lr = max_lr * (current_episode / warmup_episodes)
    else: 
        progress = (current_episode - warmup_episodes) / (total_episodes - warmup_episodes)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    return lr 


def run_episode(weights, max_steps=100, start_position=None):
    """
    the function collects data about the car with current weights 

    params:
        weights
        max steps 

    return: spike_history, reward_history, total_reward and eligibility history 

    """
    if start_position is None:
        start_position = np.random.uniform(-0.5, 0.5)

    car = CarState(position=start_position) # start centered 
    
    spike_history = []
    reward_history = []
    eligibility_history = []

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

        step_eligibility = compute_stdp_eligibility(spikes)
        eligibility_history.append(step_eligibility)

        # get action from SNN output 
        action = classify_actions(pathway_history)
        
        # Add exploration noise during training (10% random actions)
        if np.random.random() < 0.1:
            action = np.random.choice([-1, 0, 1])  # Random action

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
    eligibility_history = np.array(eligibility_history) # Shape: (T, N, N)
    
    return spike_history, reward_history, total_reward, eligibility_history


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
    
    # Early stopping
    patience = 100  # Stop if no improvement for 100 episodes
    episodes_without_improvement = 0

    # loop through episodes 
    for episode in range(num_episodes):
        # 1. run episode 
        spike_history, reward_history, total_reward, eligibility_history = run_episode(W)

        # Calculate current learning rate with warmup + cosine annealing
        ### NEED FIX ### 
        current_lr = lr_scheduler(
            current_episode=episode,
            total_episodes=num_episodes,
            warmup_episodes=50,
            max_lr=learning_rate,
            min_lr=0.0001
        )

        # 2. update weights AND baseline 
        W, baseline = apply_dopamine(
            W, 
            eligibility_history, 
            reward_history, 
            baseline, 
            baseline_lr=baseline_lr, 
            learning_rate=current_lr
        )

        # 3. append to rewards 
        episode_rewards.append(total_reward)
        
        # 4. track best weights
        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = W.copy()
            episodes_without_improvement = 0  # Reset counter
            print(f" New best! Episode {episode}: reward={total_reward:.2f}")
        else:
            episodes_without_improvement += 1
        
        # Early stopping check
        if episodes_without_improvement >= patience:
            print(f"\n Early stopping at episode {episode} (no improvement for {patience} episodes)")
            break
        
        # 6. print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}: reward={total_reward:.2f}, baseline={baseline:.2f}")
    
    print(f"\n Returning BEST weights (from episode with reward={best_reward:.2f})")
    return best_weights, episode_rewards


def train_snn_curriculum(num_episodes=500, learning_rate=0.01, baseline_lr=0.1):
    """
    progressively train SNN to drive from easy and gradually gets harder 

    beginner: 
    intermediate: 
    advanced: 
    """
    # initialization 
    W = create_weight_matrix(100)
    baseline = 0.0
    episode_rewards = []
    best_reward = -np.inf
    best_weights = W.copy()

    # 3 phase curriculum 
    curriculum = {
        1: {'name': 'easy', 'start_range': (0.0, 0.0)}, 
        2: {'name': 'medium', 'start_range': (-0.1, 0.1)}, 
        3: {'name': 'hard', 'start_range': (-0.5, 0.5)}, 
        4: {'name': 'expert', 'start_range': (-1.0, 1.0)}, 
    }

    current_phase = 1
    phase_rewards = [] # track last 20 rewards to decide when to advance 

    # run through episodes 
    for episode in range(num_episodes):
        pos_min, pos_max = curriculum[current_phase]['start_range']
        if pos_min == pos_max:
            start_pos = pos_min # always center 
        else:
            start_pos = np.random.uniform(pos_min, pos_max)

        spike_history, reward_history, total_reward, eligibility_history = run_episode(W, start_position=start_pos)

        # update weights and baseline 
        current_lr = lr_scheduler(
            current_episode=episode,
            total_episodes=num_episodes,
            warmup_episodes=100,
            max_lr=learning_rate,
            min_lr=0.001
        )
        
        W, baseline = apply_dopamine(
            W, 
            eligibility_history, 
            reward_history, 
            baseline, 
            baseline_lr=baseline_lr, 
            learning_rate=current_lr
        )

        # append to rewards 
        episode_rewards.append(total_reward)

        # track phase performance 
        phase_rewards.append(total_reward)

            # if len is more than 20, reevaluate 
        if current_phase < 4 and len(phase_rewards) >= 20:
            avg_reward = np.mean(phase_rewards[-20:])
            if current_phase == 1 and avg_reward > 50:
                print(f"\n🎉 Phase {current_phase} mastered! Advancing to phase {current_phase + 1}")
                current_phase = 2
                phase_rewards = []
            elif current_phase == 2 and avg_reward > 30:
                print(f"\n🎉 Phase {current_phase} mastered! Advancing to phase {current_phase + 1}")
                current_phase = 3 
                phase_rewards = []
            elif current_phase == 3 and avg_reward > 40:
                print(f"\n🎉 Phase {current_phase} mastered! Advancing to phase {current_phase + 1}")
                current_phase = 4
                phase_rewards = []

        # print progress every 10 episodes 
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} [Phase {current_phase}]: reward={total_reward:.2f}, baseline={baseline:.2f}")

        # track best weights 
        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = W.copy()
            print(f"New best! Episode {episode}: reward={total_reward:.2f}")

    print(f"\n✅ Returning BEST weights (from episode with reward={best_reward:.2f})")
    return best_weights, episode_rewards

