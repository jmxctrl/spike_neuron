import numpy as np 

def classify_actions(pathway_history, pool_size=33, window=10):
    """
    Map neural pathway patterns to discrete actions 
        Params:
            Pathway_history (T N) array - synaptic currents from SNN
            pool_size: int - neuron per motor pools 
            window: int - recent timesteps to average default to 10
    returns: 
        actions: -1 (left), 0 (straight), +1 (right)
    """

    # get last window timesteps 
    if pathway_history.shape[0] < window:
        recent_activity = pathway_history 
    else: 
        recent_activity = pathway_history[-window:, :] # [row 90-99, all columns]
    
    features = recent_activity.mean(axis=0)
    # split into motor pools 
    left_pool_activity = features[0:pool_size].mean() # slice from 0 to 33, average into single number
    straight_pool_activity = features[pool_size:2*pool_size].mean()
    right_pool_activity = features[2*pool_size:].mean()

    pool_scores = [left_pool_activity, straight_pool_activity, right_pool_activity]
    winning_pool = np.argmax(pool_scores)

    # convert motor pools into actions 
    action_map = {0: -1, 1: 0, 2: 1} # -1, 0, 1 == right center left 
    action = action_map[winning_pool]

    return action 

def reward_calculator(car_state, action, prev_action=0):
    """
    Calculate reward signal based on car performance
        Params:
            car_state: CarState object with position, speed, crash status 
            action: current action (-1, 0, 1)
            prev_action: previous action 
    Returns: 
        reward: float(-1000, +1.0)
    """
    
    # check catastrophic failure 
    if car_state.is_crashed():
        return -1000.0

    # check center reward 
    centering_error = car_state.get_centering()

    # position = 0.0 -> centering_error 0.0 (centered, no error) -> centering score = 1.0 (High)
    # position = 0.5 -> centering error = 0.5 (halfway to edge) -> centerng score = 0.5 (Medium)
    centering_score = 1.0 - centering_error 
    reward = centering_score * 0.5 

    # add reward for forward progress bonus 
    forward_score = car_state.speed * 0.1 
    reward += forward_score

    # penalize jerky steering 
    action_change = abs(action - prev_action)
    reward -= action_change * 0.05 

    return reward 
    
def apply_dopamine(
    weights, 
    eligibility_traces, 
    reward,
    baseline=0.0, 
    learning_rate=0.01):
    """
    Apply dopamine-modulated plasticity to synaptic weights
    Params:
        weights: (N, N) array - current synaptic weight matrix
        eligibility_traces: (N, N) array - was this synapse recently active / which synapse gets credit when reward arrives 
        reward: float - reward signal from environment
        baseline: float - expected reward (for calculating prediction error)
        learning_rate: float - learning rate scaling factor
    Returns:
        updated_weights: (N, N) array
        dopamine: float - reward prediction error signal
    """

    # Edge case 1: Shape mismatch
    assert weights.shape == eligibility_traces.shape, \
        f"Shape mismatch: weights {weights.shape} vs eligibility {eligibility_traces.shape}"
    
    # Edge case 2: Invalid reward values
    assert np.isfinite(reward), f"Reward is not finite: {reward}"
    
    # Calculate dopamine (RPE)
    dopamine = reward - baseline 
    
    # Three-factor learning rule
    weight_update = eligibility_traces * dopamine * learning_rate 
    
    # Edge case 3: Clip weights to prevent explosion
    updated_weights = np.clip(weights + weight_update, 0.0, 10.0)
    
    return updated_weights, dopamine 
    

def compute_eligibility_traces(spike_history, decay_rate=0.95):
    """
    compute eligibility traces from the spike histories, of which neuron
        Params: 
            spike_history: (T, N) array - spike data over time
            decay_rate: float - how fast traces decay (default 0.95)
    Returns:
        eligibility: (N, N) array - final eligibility trace matrix
    """
    T, N = spike_history.shape # T timesteps, N neurons 
    eligibility = np.zeros((N, N))

    for t in range(T):
        spikes = spike_history[t, :]
        pre_post_product = spikes[:, np.newaxis] * spikes[np.newaxis, :]
        #shape: (N, 1) * (1, N) = (N, N)

        eligibility += pre_post_product 
        eligibility *= decay_rate 

    return eligibility
