"""
Tests for neuromodulation.py functions

Run with: python3 test_neuromodulation.py
"""
import numpy as np
from neuromodulation import apply_dopamine, classify_actions, reward_calculator, compute_eligibility_traces


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


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" RUNNING NEUROMODULATION TESTS")
    print("="*70)
    
    try:
        test_apply_dopamine_basic()
        test_compute_eligibility_traces()
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
