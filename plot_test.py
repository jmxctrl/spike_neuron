import matplotlib.pyplot as plt
import numpy as np
import unittest
from spike_vectorized import create_weight_matrix
from spike_encode_1 import rate_encode, rate_decode

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Sine wave")
plt.show()

rng = np.random.default_rng(0) #random number generator 
samples = rng.normal(1, 1, size=1000) ## mea μ, stdσ, size= 

plt.figure()
plt.hist(samples, bins=30)
plt.title("Histogram")
plt.show()

# ===== Unit Tests for Weight Matrix =====

class TestWeightMatrix(unittest.TestCase):
    
    def test_weight_initialization(self):
        """Test that weights are initialized with correct excitatory/inhibitory ratio"""
        N = 100
        
        # Use the function from spike_vectorized.py
        W = create_weight_matrix(N, inhibitory_ratio=0.2, seed=42)
        
        # Test shape
        self.assertEqual(W.shape, (N, N))
        
        # Count excitatory and inhibitory
        excitatory_count = np.sum(W > 0)
        inhibitory_count = np.sum(W < 0)
        total = N * N
        
        # Test that all weights are accounted for
        self.assertEqual(excitatory_count + inhibitory_count, total)
        
        # Test percentage (allow 5% margin for randomness)
        inhibitory_percent = (inhibitory_count / total) * 100
        self.assertAlmostEqual(inhibitory_percent, 20.0, delta=5.0)
        
        print(f"\n✓ Test passed: {excitatory_count} excitatory, {inhibitory_count} inhibitory")
        
    def test_weight_ranges(self):
        """Test that weights are in expected ranges"""
        N = 50
        
        # Use the function from spike_vectorized.py
        W = create_weight_matrix(N, inhibitory_ratio=0.2, seed=123)
        
        # Excitatory weights should be between 0 and 1
        excitatory_weights = W[W > 0]
        self.assertTrue(np.all(excitatory_weights >= 0))
        self.assertTrue(np.all(excitatory_weights <= 1))
        
        # Inhibitory weights should be between -1 and 0
        inhibitory_weights = W[W < 0]
        self.assertTrue(np.all(inhibitory_weights >= -1))
        self.assertTrue(np.all(inhibitory_weights <= 0))
        
        print("✓ Test passed: All weights in correct ranges")

class TestSpikeEncoding(unittest.TestCase):
    
    def test_rate_encode_output_shape(self):
        """Test that rate_encode returns correct number of timesteps"""
        x = 0.5
        T = 200
        spikes = rate_encode(x, T=T, p_max=0.2)
        
        self.assertEqual(len(spikes), T)
        print(f"✓ Test passed: Encoded {T} timesteps")
    
    def test_rate_encode_output_binary(self):
        """Test that rate_encode returns only 0s and 1s"""
        x = 0.5
        spikes = rate_encode(x, T=200, p_max=0.2)
        
        self.assertTrue(np.all((spikes == 0) | (spikes == 1)))
        print("✓ Test passed: Output is binary (0s and 1s)")
    
    def test_rate_decode_inverse(self):
        """Test that rate_decode approximately recovers the original value"""
        x_original = 0.7
        spikes = rate_encode(x_original, T=1000, p_max=0.2)
        x_decoded = rate_decode(spikes, p_max=0.2)
        
        # Allow 10% error margin due to randomness
        self.assertAlmostEqual(x_decoded, x_original, delta=0.1)
        print(f"✓ Test passed: Original={x_original}, Decoded={x_decoded:.3f}")
    
    def test_encode_decode_extreme_values(self):
        """Test encoding/decoding with extreme values (0 and 1)"""
        # Test x=0 (should produce no spikes)
        spikes_zero = rate_encode(0.0, T=200, p_max=0.2)
        self.assertTrue(np.sum(spikes_zero) < 5)  # Allow a few random spikes
        
        # Test x=1 (should produce many spikes)
        spikes_one = rate_encode(1.0, T=200, p_max=0.2)
        self.assertTrue(np.sum(spikes_one) > 20)  # Should have significant spikes
        
        print("✓ Test passed: Extreme values (0 and 1) encode correctly")

if __name__ == '__main__':
    # Run unit tests
    print("\n=== Running Unit Tests ===")
    unittest.main(argv=[''], exit=False, verbosity=2)