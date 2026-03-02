import os
os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from light_task import CarState, get_sensor_values, encode_sensors_to_spikes, update_position, is_crashed
from train import load_weights, list_saved_weights
from spike_vectorized import run_vectorized_lif
from neuromodulation import classify_actions, reward_calculator

# Page config
st.set_page_config(page_title="Self-Driving Car", layout="wide", page_icon="🚗")

# Tabs for different modes
tab1, tab2 = st.tabs(["🎮 Manual Control", "🤖 AI Demo"])

# Initialize session state
if 'position' not in st.session_state:
    st.session_state.position = 0.0  # lateral position (left/right)
if 'forward_pos' not in st.session_state:
    st.session_state.forward_pos = 0.0  # forward position (up the road)
if 'speed' not in st.session_state:
    st.session_state.speed = 0.1
if 'spike_history' not in st.session_state:
    st.session_state.spike_history = []  # Store recent spike data
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0

# Sidebar controls
with st.sidebar:
    st.header("🎮 Controls")
    st.markdown("**Click buttons to drive:**")
    
    # Up button (centered)
    col_up1, col_up2, col_up3 = st.columns([1, 1, 1])
    with col_up2:
        if st.button("⬆️ UP", width='stretch', key="up"):
            st.session_state.forward_pos += st.session_state.speed * 2
            st.session_state.step_count += 1
            st.rerun()
    
    # Left and Right buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("◀️ LEFT", width='stretch', key="left"):
            st.session_state.position = update_position(st.session_state.position, -1, st.session_state.speed)
            st.session_state.step_count += 1
            st.rerun()
    
    with col3:
        if st.button("RIGHT ▶️", width='stretch', key="right"):
            st.session_state.position = update_position(st.session_state.position, 1, st.session_state.speed)
            st.session_state.step_count += 1
            st.rerun()
    
    st.divider()
    
    st.session_state.speed = st.slider("Speed", 0.02, 0.3, 0.1, 0.02)
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.position = 0.0
        st.session_state.forward_pos = 0.0
        st.session_state.spike_history = []
        st.session_state.step_count = 0
        st.rerun()
    
    st.divider()
    st.header("📊 Info")
    
    position = st.session_state.position
    forward_pos = st.session_state.forward_pos
    sensors = get_sensor_values(position)
    crashed = is_crashed(position)
    
    st.metric("Lateral Position", f"{position:.2f}")
    st.metric("Distance Traveled", f"{forward_pos:.1f}")
    
    if crashed:
        st.error("💥 CRASHED!")
    else:
        st.success("✅ On Road")
    
    st.divider()
    st.subheader("Sensors")
    st.metric("Left", f"{sensors[0]:.2f}")
    st.metric("Center", f"{sensors[1]:.2f}")
    st.metric("Right", f"{sensors[2]:.2f}")

# Main display - Full screen road
position = st.session_state.position
sensors = get_sensor_values(position)
crashed = is_crashed(position)

# Main display - Full screen road
position = st.session_state.position
forward_pos = st.session_state.forward_pos
sensors = get_sensor_values(position)
crashed = is_crashed(position)

# Generate current spike sample (single timestep)
current_spikes = encode_sensors_to_spikes(sensors, T=1, p_max=0.2).flatten()

# Store in history (keep last 50 steps)
st.session_state.spike_history.append({
    'step': st.session_state.step_count,
    'spikes': current_spikes.copy(),
    'sensors': sensors.copy()
})
if len(st.session_state.spike_history) > 50:
    st.session_state.spike_history.pop(0)

# Create layout: road on left, neurons on right
col_road, col_neurons = st.columns([1.5, 1])

with col_road:
    # Create VERTICAL road visualization (smaller size)
    fig, ax = plt.subplots(figsize=(6, 10))

    # Road parameters
    road_length = 15  # vertical length
    road_left = -1.0
    road_right = 1.0
    road_center = 0.0

    # Calculate visible road segment (scrolling effect)
    view_height = road_length
    road_start = forward_pos
    road_end = forward_pos + view_height

    # Background (grass on sides)
    ax.add_patch(patches.Rectangle((road_left - 1, road_start), 1, view_height, 
                                    facecolor='#2d5016', edgecolor='none'))
    ax.add_patch(patches.Rectangle((road_right, road_start), 1, view_height, 
                                    facecolor='#2d5016', edgecolor='none'))

    # Road surface
    ax.add_patch(patches.Rectangle((road_left, road_start), road_right - road_left, view_height, 
                                    facecolor='#2d2d2d', edgecolor='none'))

    # Lane markers (dashed white lines) - vertical with motion effect
    # Offset based on forward position to create scrolling effect
    marker_offset = forward_pos % 0.8  # Creates repeating pattern
    for y in np.arange(road_start - marker_offset, road_end, 0.8):
        ax.plot([road_center, road_center], [y, y + 0.4], 'w--', linewidth=4, alpha=0.8)
    
    # Add roadside markers/posts for motion reference
    for y in np.arange(road_start - (forward_pos % 3), road_end, 3):
        # Left side posts
        ax.plot([road_left - 0.15, road_left - 0.15], [y - 0.1, y + 0.1], 
                color='orange', linewidth=6, solid_capstyle='round', alpha=0.9)
        # Right side posts
        ax.plot([road_right + 0.15, road_right + 0.15], [y - 0.1, y + 0.1], 
                color='orange', linewidth=6, solid_capstyle='round', alpha=0.9)
    
    # Add trees/scenery objects for depth perception
    tree_positions = [-1.5, 1.5]  # Left and right sides
    for tree_x in tree_positions:
        for y in np.arange(road_start - (forward_pos % 4), road_end, 4):
            # Tree trunk
            ax.add_patch(patches.Rectangle((tree_x - 0.08, y - 0.15), 0.16, 0.3,
                                          facecolor='#3d2817', edgecolor='none'))
            # Tree foliage
            ax.add_patch(patches.Circle((tree_x, y + 0.2), 0.25,
                                       facecolor='#2d5016', edgecolor='#1a3009', linewidth=1))

    # Road edges (solid white lines) - vertical
    ax.plot([road_left, road_left], [road_start, road_end], 'white', linewidth=5)
    ax.plot([road_right, road_right], [road_start, road_end], 'white', linewidth=5)

    # Danger zones (if close to edge)
    if crashed:
        ax.add_patch(patches.Rectangle((road_left - 0.5, road_start), 0.5, view_height, 
                                        facecolor='red', alpha=0.3, edgecolor='none'))
        ax.add_patch(patches.Rectangle((road_right, road_start), 0.5, view_height, 
                                        facecolor='red', alpha=0.3, edgecolor='none'))

    # Draw car
    car_x = position  # horizontal position (lateral)
    car_y = forward_pos + 3  # vertical position on screen (always near bottom of view)
    car_width = 0.35
    car_height = 0.65

    # Car body (main)
    car_color = '#ff4444' if crashed else '#2E86DE'
    car_body = patches.Rectangle((car_x - car_width/2, car_y - car_height/2), 
                                 car_width, car_height, 
                                 facecolor=car_color, edgecolor='#1e5799', linewidth=2.5, zorder=3)
    ax.add_patch(car_body)
    
    # Car roof (slightly narrower)
    roof_width = car_width * 0.8
    roof_height = car_height * 0.35
    roof = patches.Rectangle((car_x - roof_width/2, car_y + 0.05), 
                             roof_width, roof_height, 
                             facecolor=car_color, edgecolor='#1e5799', linewidth=2, zorder=3)
    ax.add_patch(roof)

    # Windshield (front)
    windshield = patches.Polygon([
        [car_x - roof_width/2, car_y + 0.05],
        [car_x + roof_width/2, car_y + 0.05],
        [car_x + car_width/2 * 0.7, car_y - 0.05],
        [car_x - car_width/2 * 0.7, car_y - 0.05]
    ], facecolor='#B8E6FF', edgecolor='white', linewidth=1.5, alpha=0.9, zorder=4)
    ax.add_patch(windshield)
    
    # Rear window
    rear_window = patches.Rectangle((car_x - roof_width/2 * 0.7, car_y + 0.15), 
                                    roof_width * 0.7, roof_height * 0.5, 
                                    facecolor='#B8E6FF', edgecolor='white', linewidth=1, alpha=0.9, zorder=4)
    ax.add_patch(rear_window)

    # Wheels
    wheel_color = '#2C3E50'
    wheel_radius = 0.06
    # Left wheels
    left_wheel1 = patches.Circle((car_x - car_width/2 - 0.02, car_y - car_height/2 + 0.1), 
                                 wheel_radius, facecolor=wheel_color, edgecolor='black', linewidth=1.5, zorder=5)
    left_wheel2 = patches.Circle((car_x - car_width/2 - 0.02, car_y + car_height/2 - 0.1), 
                                 wheel_radius, facecolor=wheel_color, edgecolor='black', linewidth=1.5, zorder=5)
    # Right wheels
    right_wheel1 = patches.Circle((car_x + car_width/2 + 0.02, car_y - car_height/2 + 0.1), 
                                  wheel_radius, facecolor=wheel_color, edgecolor='black', linewidth=1.5, zorder=5)
    right_wheel2 = patches.Circle((car_x + car_width/2 + 0.02, car_y + car_height/2 - 0.1), 
                                  wheel_radius, facecolor=wheel_color, edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(left_wheel1)
    ax.add_patch(left_wheel2)
    ax.add_patch(right_wheel1)
    ax.add_patch(right_wheel2)
    
    # Headlights
    if not crashed:
        headlight1 = patches.Circle((car_x - car_width/3, car_y - car_height/2 - 0.02), 
                                    0.03, facecolor='#FFEB3B', edgecolor='white', linewidth=1, zorder=5)
        headlight2 = patches.Circle((car_x + car_width/3, car_y - car_height/2 - 0.02), 
                                    0.03, facecolor='#FFEB3B', edgecolor='white', linewidth=1, zorder=5)
        ax.add_patch(headlight1)
        ax.add_patch(headlight2)

    # Sensor beams
    beam_length = 2.0
    alpha_beam = 0.4

    # Left sensor beam (to left edge)
    left_intensity = sensors[0]
    if left_intensity > 0.1:
        ax.plot([car_x, road_left], [car_y, car_y], 
                color='#FF6B6B', linewidth=3 + left_intensity * 5, alpha=alpha_beam + left_intensity * 0.4)
        ax.scatter([road_left], [car_y], 
                  s=200 + left_intensity * 500, color='#FF6B6B', alpha=0.6, edgecolor='white', linewidth=2)

    # Center sensor beam (forward)
    center_intensity = sensors[1]
    if center_intensity > 0.1:
        ax.plot([car_x, car_x], [car_y, car_y + beam_length], 
                color='#4ECDC4', linewidth=3 + center_intensity * 5, alpha=alpha_beam + center_intensity * 0.4)
        ax.scatter([car_x], [car_y + beam_length], 
                  s=200 + center_intensity * 500, color='#4ECDC4', alpha=0.6, edgecolor='white', linewidth=2)

    # Right sensor beam (to right edge)
    right_intensity = sensors[2]
    if right_intensity > 0.1:
        ax.plot([car_x, road_right], [car_y, car_y], 
                color='#45B7D1', linewidth=3 + right_intensity * 5, alpha=alpha_beam + right_intensity * 0.4)
        ax.scatter([road_right], [car_y], 
                  s=200 + right_intensity * 500, color='#45B7D1', alpha=0.6, edgecolor='white', linewidth=2)
    
    # Add speed lines when moving forward (motion blur effect) - AFTER car is drawn
    if forward_pos > 0.5:  # Only show if car has moved
        speed_intensity = min(st.session_state.speed * 3, 1.0)
        num_lines = 8
        for i in range(num_lines):
            x_offset = (i - num_lines/2) * 0.4
            y_start = car_y + 5
            y_end = car_y + 8
            ax.plot([x_offset, x_offset], [y_start, y_end],
                   color='white', linewidth=1, alpha=0.1 * speed_intensity)

    # Styling
    ax.set_xlim(-2, 2)
    ax.set_ylim(road_start, road_end)
    ax.set_facecolor('#1e3a1e')
    fig.patch.set_facecolor('#0a1f0a')
    ax.axis('off')

    # Title overlay
    title_text = "💥 CRASHED!" if crashed else "🚗 Self-Driving Car"
    title_color = 'red' if crashed else 'white'
    ax.text(0, road_end - 1, title_text, 
            fontsize=24, color=title_color, weight='bold', ha='center')

    # Stats overlay
    stats_text = f"Distance: {forward_pos:.1f}m | Position: {position:.2f}"
    ax.text(0, road_start + 0.8, stats_text, 
            fontsize=12, color='white', ha='center', 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Display
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Neuron visualization panel
with col_neurons:
    st.markdown("### ⚡ Live Neuron Activity")
    
    # Current spike status for each neuron
    neuron_names = ['🔴 Left', '🟢 Center', '🔵 Right']
    neuron_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Display neurons with activity indicators
    for i, (name, color) in enumerate(zip(neuron_names, neuron_colors)):
        is_spiking = current_spikes[i] == 1
        col_n1, col_n2 = st.columns([1, 2])
        
        with col_n1:
            # Neuron visual indicator
            if is_spiking:
                st.markdown(f"<div style='width:60px; height:60px; border-radius:50%; background-color:{color}; box-shadow: 0 0 20px {color}; border: 3px solid white;'></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='width:60px; height:60px; border-radius:50%; background-color:{color}; opacity:0.3; border: 2px solid gray;'></div>", unsafe_allow_html=True)
        
        with col_n2:
            st.markdown(f"**{name}**")
            st.metric("Sensor", f"{sensors[i]:.2f}")
            if is_spiking:
                st.markdown("**⚡ FIRING!**")
    
    st.divider()
    
    # Spike history raster plot
    st.markdown("### 📊 Spike History")
    
    if len(st.session_state.spike_history) > 1:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        # Extract spike data from history
        for idx, entry in enumerate(st.session_state.spike_history):
            for neuron_idx in range(3):
                if entry['spikes'][neuron_idx] == 1:
                    ax2.scatter(idx, neuron_idx, marker='|', s=300, 
                              color=neuron_colors[neuron_idx], alpha=0.9, linewidths=3)
        
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Left', 'Center', 'Right'])
        ax2.set_xlabel('Steps', fontsize=8)
        ax2.set_ylabel('Neuron', fontsize=8)
        ax2.set_title('Recent Spikes', fontsize=10)
        ax2.set_xlim(0, 50)
        ax2.grid(True, alpha=0.2, axis='x')
        ax2.set_facecolor('#f8f8f8')
        
        st.pyplot(fig2)
        plt.close()
    else:
        st.info("Drive to see spike history!")
    
    st.divider()
    
    # Firing rate over time
    st.markdown("### 📈 Firing Rates")
    
    if len(st.session_state.spike_history) > 1:
        # Calculate rolling firing rates (spikes per 10 steps)
        window = 10
        rates = {0: [], 1: [], 2: []}
        
        for i in range(len(st.session_state.spike_history)):
            start = max(0, i - window)
            window_data = st.session_state.spike_history[start:i+1]
            
            for neuron_idx in range(3):
                spike_count = sum(1 for entry in window_data if entry['spikes'][neuron_idx] == 1)
                rate = spike_count / len(window_data)
                rates[neuron_idx].append(rate)
        
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        
        for neuron_idx in range(3):
            ax3.plot(rates[neuron_idx], color=neuron_colors[neuron_idx], 
                    linewidth=2, label=neuron_names[neuron_idx], alpha=0.8)
        
        ax3.set_xlabel('Steps', fontsize=8)
        ax3.set_ylabel('Firing Rate', fontsize=8)
        ax3.set_title('Neuron Activity Over Time', fontsize=10)
        ax3.legend(loc='upper right', fontsize=7)
        ax3.grid(True, alpha=0.2)
        ax3.set_ylim(0, 0.3)
        ax3.set_facecolor('#f8f8f8')
        
        st.pyplot(fig3)
        plt.close()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔴 Left Sensor")
    st.progress(float(sensors[0]))
    st.metric("Value", f"{sensors[0]:.3f}")

with col2:
    st.markdown("### 🟢 Center Sensor")
    st.progress(float(sensors[1]))
    st.metric("Value", f"{sensors[1]:.3f}")

with col3:
    st.markdown("### 🔵 Right Sensor")
    st.progress(float(sensors[2]))
    st.metric("Value", f"{sensors[2]:.3f}")

# Spike encoding visualization
with st.expander("⚡ View Spike Encoding", expanded=False):
    T = st.slider("Timesteps", 50, 300, 100, 10)
    p_max = st.slider("Max Spike Probability", 0.1, 0.5, 0.2, 0.05)
    
    spikes = encode_sensors_to_spikes(sensors, T=T, p_max=p_max)
    
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    
    labels = ['Left', 'Center', 'Right']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i in range(3):
        spike_times = np.where(spikes[i] == 1)[0]
        ax2.scatter(spike_times, [i] * len(spike_times), 
                   marker='|', s=200, color=colors[i], alpha=0.8, linewidths=2)
    
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Sensor', fontsize=12)
    ax2.set_title('Neural Spike Trains', fontsize=14)
    ax2.set_xlim(0, T)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_facecolor('#f0f0f0')
    
    st.pyplot(fig2)
    plt.close()
    
    st.info(f"**Spike counts:** Left={spikes[0].sum()}, Center={spikes[1].sum()}, Right={spikes[2].sum()}")
