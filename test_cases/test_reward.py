"""Test reward_calculator function with various scenarios"""
import sys
sys.path.append('/Users/chubee/Documents/Neuro-Encoder')

from light_task import CarState
from neuromodulation import reward_calculator

def assert_close(actual, expected, test_name, tolerance=0.001):
    """Check if actual value is close to expected value"""
    if abs(actual - expected) <= tolerance:
        print(f"   ✓ PASS: {test_name}")
        return True
    else:
        print(f"   ✗ FAIL: {test_name}")
        print(f"      Expected: {expected:.3f}, Got: {actual:.3f}, Diff: {abs(actual - expected):.3f}")
        return False

print("=" * 60)
print("TESTING REWARD FUNCTION WITH ASSERTIONS")
print("=" * 60)

passed = 0
total = 0

# Test 1: Crashed car
print("\n1. CRASHED CAR:")
crashed_car = CarState(position=1.5, speed=1.0)
reward = reward_calculator(crashed_car, action=0, prev_action=0)
expected = -1000.0
print(f"   Position: {crashed_car.position}, Crashed: {crashed_car.is_crashed()}")
print(f"   Expected: {expected:.2f}, Got: {reward:.2f}")
total += 1
if assert_close(reward, expected, "Crash penalty"):
    passed += 1

# Test 2: Perfect centering, good speed, smooth steering
print("\n2. PERFECT PERFORMANCE:")
perfect_car = CarState(position=0.0, speed=1.0)
reward = reward_calculator(perfect_car, action=0, prev_action=0)
# Centering: (1.0 - 0.0) * 0.5 = 0.5
# Forward: 1.0 * 0.1 = 0.1
# Smoothness: 0
expected = 0.5 + 0.1
print(f"   Position: {perfect_car.position}, Speed: {perfect_car.speed}")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Perfect performance"):
    passed += 1

# Test 3: Off-center position
print("\n3. OFF-CENTER (position=0.3):")
offcenter_car = CarState(position=0.3, speed=1.0)
reward = reward_calculator(offcenter_car, action=0, prev_action=0)
centering_error = offcenter_car.get_centering()
# Centering: (1.0 - 0.3) * 0.5 = 0.35
# Forward: 1.0 * 0.1 = 0.1
# Smoothness: 0
expected = (1.0 - centering_error) * 0.5 + 0.1
print(f"   Position: {offcenter_car.position}, Centering error: {centering_error:.3f}")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Off-center position"):
    passed += 1

# Test 4: Near edge (almost crashing)
print("\n4. NEAR EDGE (position=0.49):")
edge_car = CarState(position=0.49, speed=1.0)
reward = reward_calculator(edge_car, action=0, prev_action=0)
centering_error = edge_car.get_centering()
# Centering: (1.0 - 0.49) * 0.5 = 0.255
# Forward: 1.0 * 0.1 = 0.1
# Smoothness: 0
expected = (1.0 - centering_error) * 0.5 + 0.1
print(f"   Position: {edge_car.position}, Crashed: {edge_car.is_crashed()}")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Near edge"):
    passed += 1

# Test 5: Jerky steering (straight → right)
print("\n5. JERKY STEERING (straight → right):")
jerky_car = CarState(position=0.0, speed=1.0)
reward = reward_calculator(jerky_car, action=1, prev_action=0)
# Centering: 0.5
# Forward: 0.1
# Smoothness: -1 * 0.05 = -0.05
expected = 0.5 + 0.1 - 0.05
print(f"   Action change: 1 (0 → 1)")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Jerky steering"):
    passed += 1

# Test 6: Very jerky steering (left → right)
print("\n6. VERY JERKY STEERING (left → right):")
very_jerky_car = CarState(position=0.0, speed=1.0)
reward = reward_calculator(very_jerky_car, action=1, prev_action=-1)
# Centering: 0.5
# Forward: 0.1
# Smoothness: -2 * 0.05 = -0.1
expected = 0.5 + 0.1 - 0.1
print(f"   Action change: 2 (-1 → 1)")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Very jerky steering"):
    passed += 1

# Test 7: Slow speed
print("\n7. SLOW SPEED:")
slow_car = CarState(position=0.0, speed=0.3)
reward = reward_calculator(slow_car, action=0, prev_action=0)
# Centering: 0.5
# Forward: 0.3 * 0.1 = 0.03
# Smoothness: 0
expected = 0.5 + 0.03
print(f"   Speed: {slow_car.speed}")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Slow speed"):
    passed += 1

# Test 8: Fast speed
print("\n8. FAST SPEED:")
fast_car = CarState(position=0.0, speed=2.0)
reward = reward_calculator(fast_car, action=0, prev_action=0)
# Centering: 0.5
# Forward: 2.0 * 0.1 = 0.2
# Smoothness: 0
expected = 0.5 + 0.2
print(f"   Speed: {fast_car.speed}")
print(f"   Expected: {expected:.3f}, Got: {reward:.3f}")
total += 1
if assert_close(reward, expected, "Fast speed"):
    passed += 1

# =============================================================================
# RANDOMIZED STRESS TESTS - Check properties across many random scenarios
# =============================================================================
print("\n" + "=" * 60)
print("RANDOMIZED STRESS TESTS (100 iterations)")
print("=" * 60)

import random
random.seed(42)  # For reproducibility

stress_passed = 0
stress_total = 0

for i in range(100):
    # Random parameters
    position = random.uniform(-1.5, 1.5)  # Can be outside bounds
    speed = random.uniform(0.0, 3.0)
    action = random.choice([-1, 0, 1])
    prev_action = random.choice([-1, 0, 1])
    
    car = CarState(position=position, speed=speed)
    reward = reward_calculator(car, action=action, prev_action=prev_action)
    
    # Property 1: Crashed cars always get -1000
    if car.is_crashed():
        stress_total += 1
        if reward == -1000.0:
            stress_passed += 1
        else:
            print(f"   ✗ FAIL: Crashed car at position {position:.2f} got reward {reward:.2f} instead of -1000.0")
    
    # Property 2: Non-crashed rewards should be in reasonable range
    else:
        stress_total += 1
        # Max possible: centering=0.5, forward=0.3, smoothness=0 ≈ 0.8
        # Min possible: centering=0.0, forward=0, smoothness=-0.1 ≈ -0.1
        if -0.2 <= reward <= 1.0:
            stress_passed += 1
        else:
            print(f"   ✗ FAIL: Non-crashed reward {reward:.3f} out of range [-0.2, 1.0]")
            print(f"      Position: {position:.2f}, Speed: {speed:.2f}, Actions: {prev_action}→{action}")

# Property 3: Better centering should give higher reward (holding other factors constant)
print("\n   Testing centering property...")
prop3_tests = 0
prop3_passed = 0
for _ in range(20):
    speed = random.uniform(0.5, 2.0)
    action = 0
    prev_action = 0
    
    car_centered = CarState(position=0.0, speed=speed)
    car_offcenter = CarState(position=0.5, speed=speed)
    
    reward_centered = reward_calculator(car_centered, action, prev_action)
    reward_offcenter = reward_calculator(car_offcenter, action, prev_action)
    
    prop3_tests += 1
    stress_total += 1
    if reward_centered > reward_offcenter:
        prop3_passed += 1
        stress_passed += 1
    else:
        print(f"   ✗ FAIL: Centered reward {reward_centered:.3f} not > offcenter {reward_offcenter:.3f}")

print(f"   Centering property: {prop3_passed}/{prop3_tests} passed")

# Property 4: Higher speed should give higher reward (holding other factors constant)
print("\n   Testing speed property...")
prop4_tests = 0
prop4_passed = 0
for _ in range(20):
    position = random.uniform(-0.5, 0.5)
    action = 0
    prev_action = 0
    
    car_slow = CarState(position=position, speed=0.5)
    car_fast = CarState(position=position, speed=2.0)
    
    reward_slow = reward_calculator(car_slow, action, prev_action)
    reward_fast = reward_calculator(car_fast, action, prev_action)
    
    prop4_tests += 1
    stress_total += 1
    if reward_fast > reward_slow:
        prop4_passed += 1
        stress_passed += 1
    else:
        print(f"   ✗ FAIL: Fast reward {reward_fast:.3f} not > slow {reward_slow:.3f}")

print(f"   Speed property: {prop4_passed}/{prop4_tests} passed")

# Property 5: Smooth steering should give higher reward (holding other factors constant)
print("\n   Testing smoothness property...")
prop5_tests = 0
prop5_passed = 0
for _ in range(20):
    position = random.uniform(-0.5, 0.5)
    speed = random.uniform(0.5, 2.0)
    
    car = CarState(position=position, speed=speed)
    
    # Smooth: straight → straight
    reward_smooth = reward_calculator(car, action=0, prev_action=0)
    
    # Jerky: left → right
    reward_jerky = reward_calculator(car, action=1, prev_action=-1)
    
    prop5_tests += 1
    stress_total += 1
    if reward_smooth > reward_jerky:
        prop5_passed += 1
        stress_passed += 1
    else:
        print(f"   ✗ FAIL: Smooth reward {reward_smooth:.3f} not > jerky {reward_jerky:.3f}")

print(f"   Smoothness property: {prop5_passed}/{prop5_tests} passed")

print("\n" + "=" * 60)
print("STRESS TEST RESULTS")
print("=" * 60)
print(f"Passed: {stress_passed}/{stress_total} property checks")
if stress_passed == stress_total:
    print("✓ ALL PROPERTIES HOLD!")
else:
    print(f"✗ {stress_total - stress_passed} property check(s) failed")

print("\n" + "=" * 60)
print("OVERALL TEST RESULTS")
print("=" * 60)
print("\n" + "=" * 60)
print("OVERALL TEST RESULTS")
print("=" * 60)
print(f"Fixed tests: {passed}/{total}")
print(f"Stress tests: {stress_passed}/{stress_total}")
print(f"Total: {passed + stress_passed}/{total + stress_total}")
if passed == total and stress_passed == stress_total:
    print("\n✓ ALL TESTS PASSED!")
else:
    print(f"\n✗ {(total - passed) + (stress_total - stress_passed)} test(s) failed")
print("=" * 60)
