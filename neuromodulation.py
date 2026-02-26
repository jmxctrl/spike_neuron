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

def compute_stdp_eligibility(
    spike_history, 
    tau_plus=20,
    A_plus=1.0,
    decay_rate=0.95,
    ):
    """
    compute eligibility traces from the spike histories, of which neuron

        Calculation: 
            Δwij = A+ * e^-(Δt/τ) -- synaptic weight from neuron i to neuron j 
                A+ = maximum strengthening amount 
                Δt/τ = how many τ passed 
                e^(Δt/τ) = long term depression 

            eij(T) = 
                summation of alpha ^ T-t' from 0 to t
                summation of delta t - tau_plus 
                    pre_spike[i] = s_i(t - dt)
                    post_spike[j] = s_j(t)
                    STDP weights for the time gap  = Δwij

        Params: 
            spike_history: (T, N) array - spike data over time
            tau_plus: causal relationship if fired within last t timesteps 
            A_plus: how much connection to strengthen 
            decay_rate: float - how fast traces decay (default 0.95)

        Goal: 
            for each synapse i->j, calculate how much to strengthen 
                - did neuron i fire before j (causal relationship)
                - how close in time (exponential decay)
    Returns:
        eligibility: (N, N) array - final eligibility trace matrix
    """
    T, N = spike_history.shape # T timesteps, N neurons 
    eligibility = np.zeros((N, N))

    for t in range(T):
        post_spikes = spike_history[t, :] # spiked now 

        # look backwards for pre-synaptic spikes 
        for dt in range(1, tau_plus + 1):
            if t - dt >= 0:
                pre_spikes = spike_history[t-dt, :] 

                # CORE: STDP weight 
                weight = A_plus * np.exp(-dt / tau_plus)

                # weighted outer spikes and eligibility 
                update = weight * (pre_spikes[:, np.newaxis] * post_spikes[np.newaxis, :])
                eligibility += update

        eligibility *= decay_rate 

    return eligibility


def apply_dopamine(
    weights, 
    eligibility_history, 
    reward_history,
    baseline=0.0, 
    baseline_lr=0.1,
    learning_rate=0.01,
    w_min=0.0,
    w_max=10.0):
    """
    Apply dopamine-modulated plasticity to synaptic weights (batch processing)
    
    Params:
        weights: (N, N) array - initial synaptic weight matrix
        eligibility_history: (T, N, N) array - STDP eligibility traces over time
        reward_history: (T,) array - reward signals over time
        baseline: float - initial expected reward (for calculating prediction error)
            baseline_new = (1 - α) * baseline_old + α * reward 
            example: 
                90% weight to old baseline 
                10% weight to new reward
        baseline_lr: float - learning rate for baseline adaptation (default 0.1)
        learning_rate: float - learning rate scaling factor (must be >= 0)
        w_min: float - minimum weight value (default 0.0)
        w_max: float - maximum weight value (default 10.0)
    
    Returns:
        updated_weights: (N, N) array - final weights after processing all timesteps
        baseline: float - updated baseline after processing all rewards
    """

    # Validate inputs
    T = len(reward_history)
    assert len(eligibility_history) == T, \
        f"Length mismatch: eligibility_history {len(eligibility_history)} vs reward_history {T}"
    
    assert eligibility_history.shape[1:] == weights.shape, \
        f"Shape mismatch: eligibility {eligibility_history.shape[1:]} vs weights {weights.shape}"
    
    # Edge case 2: Invalid reward values
    assert np.all(np.isfinite(reward_history)), "Reward history contains NaN or Inf"
    
    # Edge case 3: Check eligibility for invalid values
    assert np.all(np.isfinite(eligibility_history)), "Eligibility history contains NaN or Inf"
    
    # Edge case 4: Validate learning rate
    assert learning_rate >= 0, f"Learning rate must be non-negative, got {learning_rate}"
    
    # Edge case 5: Validate baseline
    assert np.isfinite(baseline), f"Baseline is not finite: {baseline}"
    
    # Edge case 6: Validate weight bounds
    assert w_min < w_max, f"w_min ({w_min}) must be less than w_max ({w_max})"
    
    # Process each timestep
    T = len(reward_history)
    
    for t in range(T):
        # Calculate prediction error 
        dopamine = reward_history[t] - baseline
        
        # Update baseline (exponential moving average)
        baseline += baseline_lr * dopamine
        
        # Update weights using three-factor rule
        weight_update = eligibility_history[t] * dopamine * learning_rate
        weights = np.clip(weights + weight_update, w_min, w_max)
    
    return weights, baseline 

