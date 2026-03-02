import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import time
from train import load_weights, list_saved_weights
from light_task import CarState, get_sensor_values, encode_sensors_to_spikes
from spike_vectorized import run_vectorized_lif
from neuromodulation import classify_actions, reward_calculator

st.set_page_config(page_title="SNN Car Demo", layout="wide")

st.title("🚗 SNN Autonomous Car - Live Demo")
st.markdown("Watch your trained spiking neural network drive in real-time!")

# Sidebar for controls
st.sidebar.header("⚙️ Controls")

# List and select weights (from parent directory)
weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_weights")
weights_files = list_saved_weights(weights_dir)
if not weights_files:
    st.error("No trained weights found! Train a model first with `python3 run_training.py`")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select trained weights:",
    sorted(weights_files, reverse=True)
)

max_steps = st.sidebar.slider("Max steps per episode:", 50, 200, 100)
speed = st.sidebar.slider("Simulation speed (steps/sec):", 1, 20, 10)

# Load weights
if st.sidebar.button("🚀 Start Demo") or 'running' in st.session_state:
    st.session_state.running = True
    
    # Path relative to parent directory
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_weights", selected_file)
    W = load_weights(weights_path)
    
    st.sidebar.success(f"Loaded: {selected_file}")
    
    # Initialize car
    car = CarState(position=0.0)
    prev_action = 0
    total_reward = 0
    
    # Create placeholders for live updates
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🛣️ Track View")
        track_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Metrics")
        metrics_placeholder = st.empty()
    
    with col3:
        st.subheader("🎮 Actions")
        action_placeholder = st.empty()
    
    # Spike activity visualization
    st.subheader("⚡ Neural Activity")
    spike_placeholder = st.empty()
    
    # Run simulation
    pathway_history = []
    step = 0
    
    while step < max_steps and not car.is_crashed():
        # 1. Get sensor values (enhanced=True gives 5 sensors)
        sensor_values = get_sensor_values(car, enhanced=True)
        
        # 2. Encode to spikes
        input_spikes = encode_sensors_to_spikes(sensor_values, T=10)
        
        # 3. Run SNN (returns spikes, voltages, pathway_history)
        spikes, voltages, pathway = run_vectorized_lif(
            W=W,
            external_spikes=input_spikes,
            num_steps=10,
            input_current=5.0,
            plot=False
        )
        
        pathway_history.append(pathway[-1, :])
        
        # 4. Classify action
        if len(pathway_history) >= 10:
            action = classify_actions(np.array(pathway_history[-10:]))
        else:
            action = 0
        
        # 5. Update car
        car.update(action)
        
        # 6. Calculate reward
        reward = reward_calculator(car, action, prev_action)
        total_reward += reward
        prev_action = action
        
        # Visualize track
        with track_placeholder.container():
            # Draw track boundaries
            track_width = 1.0
            position = car.position
            
            # Create ASCII-style track visualization
            track_viz = ""
            left_edge = -track_width / 2
            right_edge = track_width / 2
            
            # Create track representation
            track_chars = 50
            car_pos_idx = int((position - left_edge) / track_width * track_chars)
            car_pos_idx = max(0, min(track_chars - 1, car_pos_idx))
            
            # Create vertical track view with forward motion
            track_height = 15
            track_viz_lines = []
            
            for row in range(track_height):
                # Calculate lane marker positions (animated based on step count)
                marker_offset = step % 4  # Creates scrolling effect
                
                if row == track_height - 3:  # Car position (near bottom)
                    # Draw car at lateral position - using ▲ for upward-facing car
                    line = "║" + "".join(
                        ["▲" if i == car_pos_idx else (" " if i % 8 != marker_offset else "░") for i in range(track_chars)]
                    ) + "║"
                elif row % 4 == marker_offset:  # Lane markers
                    line = "║" + "".join(
                        ["┊" if i == track_chars//4 or i == 3*track_chars//4 else " " for i in range(track_chars)]
                    ) + "║"
                else:  # Empty road
                    line = "║" + " " * track_chars + "║"
                
                track_viz_lines.append(line)
            
            # Display track
            track_display = "\n".join(track_viz_lines)
            st.markdown(f"```\n{track_display}\n```")
            
            # Position indicator bar
            progress_val = max(0.0, min(1.0, (position + track_width/2) / track_width))
            st.progress(progress_val)
            st.caption(f"Position: {position:.3f} (Center = 0.0)")
            
            # Status
            status = "🟢 Driving" if not car.is_crashed() else "🔴 CRASHED"
            st.markdown(f"**Status:** {status}")
        
        # Display metrics
        with metrics_placeholder.container():
            st.metric("Step", step)
            st.metric("Position", f"{position:.3f}")
            st.metric("Speed", f"{car.speed:.2f}")
            st.metric("Total Reward", f"{total_reward:.1f}")
            st.metric("Instant Reward", f"{reward:.2f}")
        
        # Display action
        with action_placeholder.container():
            action_map = {-1: "⬅️ LEFT", 0: "⬆️ STRAIGHT", 1: "➡️ RIGHT"}
            st.markdown(f"### {action_map.get(action, 'UNKNOWN')}")
            
            # Show motor pool activities
            if len(pathway_history) >= 10:
                recent = np.array(pathway_history[-10:])
                left_act = recent[:, 0:33].mean()
                straight_act = recent[:, 33:66].mean()
                right_act = recent[:, 66:].mean()
                
                st.markdown("**Motor Pools:**")
                st.progress(float(left_act / max(left_act, straight_act, right_act, 0.1)))
                st.caption(f"Left: {left_act:.2f}")
                st.progress(float(straight_act / max(left_act, straight_act, right_act, 0.1)))
                st.caption(f"Straight: {straight_act:.2f}")
                st.progress(float(right_act / max(left_act, straight_act, right_act, 0.1)))
                st.caption(f"Right: {right_act:.2f}")
        
        # Visualize spike activity
        with spike_placeholder.container():
            if len(pathway_history) > 0:
                recent_spikes = spikes[-1, :]  # Last timestep
                
                # Show input neurons (sensors)
                st.markdown("**Input Neurons (Sensors):**")
                sensor_viz = "".join(["🟡" if s > 0 else "⚫" for s in recent_spikes[:5]])
                st.markdown(f"`{sensor_viz}`")
                
                # Show motor neurons activity
                st.markdown("**Motor Neurons (Actions):**")
                motor_activity = pathway_history[-1]
                
                # Create heatmap-style visualization
                cols = st.columns(3)
                with cols[0]:
                    st.caption("Left")
                    left_heat = motor_activity[0:33].mean()
                    st.progress(float(min(1.0, left_heat / 10)))
                with cols[1]:
                    st.caption("Straight")
                    straight_heat = motor_activity[33:66].mean()
                    st.progress(float(min(1.0, straight_heat / 10)))
                with cols[2]:
                    st.caption("Right")
                    right_heat = motor_activity[66:].mean()
                    st.progress(float(min(1.0, right_heat / 10)))
        
        step += 1
        time.sleep(1.0 / speed)  # Control simulation speed
    
    # Final results
    st.markdown("---")
    st.subheader("📈 Episode Complete!")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric("Total Steps", step)
    with result_col2:
        st.metric("Final Reward", f"{total_reward:.2f}")
    with result_col3:
        st.metric("Final Status", "CRASHED" if car.is_crashed() else "COMPLETED")
    
    if st.button("🔄 Run Again"):
        st.session_state.running = False
        st.rerun()

else:
    st.info("👈 Select weights and click 'Start Demo' to begin!")
    
    # Show preview of available models
    st.subheader("Available Trained Models:")
    for wf in sorted(weights_files, reverse=True):
        # Extract reward from filename
        if "reward" in wf:
            reward_str = wf.split("reward")[1].split("_")[0]
            st.markdown(f"- `{wf}` - Best training reward: **{reward_str}**")
