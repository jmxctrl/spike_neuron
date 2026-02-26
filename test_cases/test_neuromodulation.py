"""
Tests for neuromodulation.py functions

Run with: python3 test_cases/test_neuromodulation.py (from Neuro-Encoder directory)
"""
import numpy as np
import sys
sys.path.insert(0, '.')  # Add current directory to path

from neuromodulation import apply_dopamine, classify_actions, reward_calculator, compute_stdp_eligibility


def test_apply_dopamine_basic():
    """Test basic dopamine modulation calculation"""
    print("\n" + "="*60)
    print("TEST: Basic dopamine modulation")
    print("="*60)
    
    weights = np.array([[0.5, 0.3],
                        [0.2, 0.4]])

    eligibility = np.array([[1.0, 0.0],
                           [0.5, 0.0]])

    reward = 1.0 
    baseline = 0.0
    learning_rate = 0.1 

    new_w, dopa = apply_dopamine(weights, eligibility, reward, baseline, learning_rate)

    print("Original weights:")
    print(weights)
    print("\nEligibility:")
    print(eligibility)
    print(f"\nReward: {reward}, Baseline: {baseline}")
    print(f"Dopamine (RPE): {dopa}")
    print("\nUpdated weights:")
    print(new_w)
    print("\nExpected: [[0.6, 0.3], [0.25, 0.4]]")
    
    # Verify dopamine calculation
    assert dopa == 1.0, f"Expected dopamine=1.0, got {dopa}"
    
    # Verify weight updates (with small tolerance for floating point)
    expected = np.array([[0.6, 0.3], [0.25, 0.4]])
    assert np.allclose(new_w, expected), f"Weight update incorrect"
    
    print("✓ Basic dopamine test passed")


def test_compute_eligibility_traces():
    """Test eligibility trace computation"""
    print("\n" + "="*60)
    print("TEST: Compute eligibility traces")
    print("="*60)

    # Toy spike data: 3 timesteps, 2 neurons
    spike_history = np.array([
        [1, 0],  # t=0: neuron 0 fires
        [1, 1],  # t=1: both fire
        [0, 1]   # t=2: neuron 1 fires
    ])

    eligibility = compute_eligibility_traces(spike_history, decay_rate=0.9)
    print("Spike history:")
    print(spike_history)
    print(f"\nFinal eligibility traces (decay=0.9):")
    print(eligibility)
    
    # Verify shape
    assert eligibility.shape == (2, 2), f"Expected shape (2,2), got {eligibility.shape}"
    
    print("✓ Eligibility trace test passed")


def test_edge_cases():
    """Test edge case handling"""
    print("\n" + "="*60)
    print("TEST: Edge cases")
    print("="*60)
    
    weights = np.array([[0.5, 0.3],
                        [0.2, 0.4]])
    
    eligibility = np.array([[1.0, 0.0],
                           [0.5, 0.0]])
    
    # Test 1: Shape mismatch error 
    try: 
        bad_elig = np.array([1.0])
        apply_dopamine(weights, bad_elig, 1.0)
        print("✗ Failed to catch shape mismatch")
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        print(f"✓ Caught shape mismatch: {e}")

    # Test 2: Infinite reward error 
    try:
        apply_dopamine(weights, eligibility, np.inf)
        print("✗ Failed to catch infinite reward")
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        print(f"✓ Caught infinite reward: {e}")
    
    # Test 3: None values
    try:
        apply_dopamine(None, eligibility, 1.0)
        print("✗ Failed to catch None weights")
        assert False, "Should have raised error"
    except (AssertionError, AttributeError) as e:
        print(f"✓ Caught None weights: {type(e).__name__}")
    
    print("✓ All edge case tests passed")


def test_negative_rpe():
    """Test negative reward prediction error (worse than expected)"""
    print("\n" + "="*60)
    print("TEST: Negative RPE (worse than expected)")
    print("="*60)
    
    weights = np.array([[1.0, 0.5],
                        [0.5, 1.0]])
    
    eligibility = np.array([[1.0, 0.5],
                           [0.5, 1.0]])
    
    reward = 0.0  # Bad outcome
    baseline = 0.5  # Expected better
    learning_rate = 0.1
    
    new_w, dopa = apply_dopamine(weights, eligibility, reward, baseline, learning_rate)
    
    print(f"Reward: {reward}, Baseline: {baseline}")
    print(f"Dopamine (RPE): {dopa}")
    print(f"Original weights: {weights[0,0]:.3f}")
    print(f"Updated weights: {new_w[0,0]:.3f}")
    
    assert dopa < 0, "Dopamine should be negative"
    assert new_w[0,0] < weights[0,0], "Weights should decrease with negative RPE"
    
    print("✓ Negative RPE test passed")


def test_stdp_eligibility():
    """Test STDP-based eligibility trace computation"""
    print("\n" + "="*60)
    print("TEST: STDP eligibility traces")
    print("="*60)
    
    # Test Case 1: Simple causal relationship
    # Neuron 0 fires at t=0, Neuron 1 fires at t=2 (2 steps later)
    spike_history = np.array([
        [1, 0],  # t=0: neuron 0 fires (PRE)
        [0, 0],  # t=1: nothing
        [0, 1],  # t=2: neuron 1 fires (POST)
        [0, 0],  # t=3: nothing
    ])
    
    eligibility = compute_stdp_eligibility(spike_history, tau_plus=5, A_plus=1.0, decay_rate=0.9)
    
    print("Spike history (0→1 causal):")
    print(spike_history)
    print(f"\nEligibility matrix:")
    print(eligibility)
    print(f"\neligibility[0,1] (0→1, causal): {eligibility[0,1]:.4f}")
    print(f"eligibility[1,0] (1→0, non-causal): {eligibility[1,0]:.4f}")
    
    # Verify: 0→1 should be stronger than 1→0
    assert eligibility[0,1] > eligibility[1,0], "Causal connection should be stronger"
    
    # Test Case 2: Check STDP timing decay
    # Close spikes should have stronger eligibility
    spike_history_close = np.array([
        [1, 0],  # t=0: pre fires
        [0, 1],  # t=1: post fires (dt=1, very close!)
    ])
    
    spike_history_far = np.array([
        [1, 0, 0, 0, 0, 0],  # t=0: pre fires
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],  # t=5: post fires (dt=5, farther)
    ])
    
    elig_close = compute_stdp_eligibility(spike_history_close, tau_plus=10, A_plus=1.0, decay_rate=1.0)
    elig_far = compute_stdp_eligibility(spike_history_far, tau_plus=10, A_plus=1.0, decay_rate=1.0)
    
    print(f"\nTiming test:")
    print(f"  Close (dt=1): eligibility[0,1] = {elig_close[0,1]:.4f}")
    print(f"  Far (dt=5):   eligibility[0,1] = {elig_far[0,1]:.4f}")
    
    # Verify exponential decay: closer spikes = stronger
    assert elig_close[0,1] > elig_far[0,1], "Closer spikes should have stronger eligibility"
    
    # Test Case 3: Check shape
    assert eligibility.shape == (2, 2), f"Expected shape (2,2), got {eligibility.shape}"
    
    print("✓ STDP eligibility test passed")


def test_custom_weight_bounds():
    """Test custom weight bounds (w_min, w_max)"""
    print("\n" + "="*60)
    print("TEST: Custom weight bounds")
    print("="*60)
    
    weights = np.ones((2, 2)) * 0.5
    eligibility = np.ones((2, 2))
    reward = 10.0  # Huge reward to trigger clipping
    baseline = 0.0
    learning_rate = 1.0  # Strong learning
    
    # Test 1: Default bounds (w_max=10.0)
    new_w_default, _ = apply_dopamine(weights, eligibility, reward, baseline, learning_rate)
    print(f"Default w_max=10.0:")
    print(f"  Original weight: {weights[0,0]:.2f}")
    print(f"  After update: {new_w_default[0,0]:.2f}")
    assert np.max(new_w_default) == 10.0, "Should clip at default w_max=10.0"
    
    # Test 2: Custom tight bounds (w_max=2.0)
    new_w_tight, _ = apply_dopamine(weights, eligibility, reward, baseline, learning_rate, w_max=2.0)
    print(f"\nCustom w_max=2.0:")
    print(f"  After update: {new_w_tight[0,0]:.2f}")
    assert np.max(new_w_tight) == 2.0, "Should clip at custom w_max=2.0"
    assert new_w_tight[0,0] < new_w_default[0,0], "Tighter bound should result in smaller weights"
    
    # Test 3: Allow negative weights (w_min=-5.0)
    weights_neg = np.ones((2, 2)) * 0.5
    eligibility_neg = np.ones((2, 2)) * -1.0  # Negative eligibility
    new_w_neg, _ = apply_dopamine(weights_neg, eligibility_neg, reward, baseline, learning_rate, 
                                   w_min=-5.0, w_max=5.0)
    print(f"\nCustom bounds w_min=-5.0, w_max=5.0:")
    print(f"  With negative eligibility: {new_w_neg[0,0]:.2f}")
    assert np.min(new_w_neg) >= -5.0, "Should respect w_min=-5.0"
    
    # Test 4: Verify w_min works
    weights_high = np.ones((2, 2)) * 3.0
    eligibility_high = np.ones((2, 2))
    reward_bad = -10.0  # Big negative reward
    new_w_min, _ = apply_dopamine(weights_high, eligibility_high, reward_bad, baseline, learning_rate,
                                   w_min=1.0, w_max=10.0)
    print(f"\nTesting w_min=1.0 with negative reward:")
    print(f"  After negative update: {new_w_min[0,0]:.2f}")
    assert np.min(new_w_min) == 1.0, "Should clip at w_min=1.0"
    
    print("✓ Custom weight bounds test passed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" RUNNING NEUROMODULATION TESTS")
    print("="*70)
    
    try:
        test_apply_dopamine_basic()
        # test_compute_eligibility_traces()  # Old test - function replaced by compute_stdp_eligibility
        test_stdp_eligibility()
        test_custom_weight_bounds()
        test_edge_cases()
        test_negative_rpe()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
    except AssertionError as e:
        print("\n" + "="*70)
        print(f"✗ TEST FAILED: {e}")
        print("="*70)
        raise
