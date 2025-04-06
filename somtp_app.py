import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# --- Streamlit interface ---
st.set_page_config(page_title="Optimized Trajectory Planner with MPC", layout="wide")
st.title("🚀 Optimized Trajectory Planner with MPC")

# --- Sidebar Header and Styling ---
st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f4f7fc;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease-in-out;
        cursor: pointer;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2575fc, #6a11cb);
        transform: scale(1.05);
    }
    .prediction-result {
        font-size: 22px;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .churned {
        background-color: #ff4b4b;
        color: white;
    }
    .retained {
        background-color: #4CAF50;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f4f7fc;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar AI-Powered Information Section ---
st.sidebar.markdown("### 🤖 AI-powered Optimized Trajectory Planner")
st.sidebar.info(
    """
    This tool uses **AI** and **MPC** to optimize your trajectory while avoiding obstacles,
    ensuring smoothness, and reaching the goal. ✨
    Adjust parameters on the left to generate the best path! 🛤️
    """
)

# --- Input Parameters ---
st.sidebar.header("⚙️ Input Parameters")

# Start and Goal sliders with tooltips
start_x = st.sidebar.slider("🟢 Start X", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Adjust the starting X position of the trajectory.")
start_y = st.sidebar.slider("🟢 Start Y", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Adjust the starting Y position of the trajectory.")
goal_x = st.sidebar.slider("🔴 Goal X", min_value=0.0, max_value=10.0, value=9.0, step=0.1, help="Adjust the goal X position of the trajectory.")
goal_y = st.sidebar.slider("🔴 Goal Y", min_value=0.0, max_value=10.0, value=9.0, step=0.1, help="Adjust the goal Y position of the trajectory.")

# Trajectory settings sliders
n_waypoints = st.sidebar.slider("🔹 Number of Waypoints", min_value=5, max_value=50, value=20, help="Set the number of waypoints in the trajectory.")
lr = st.sidebar.slider("🔸 Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, help="Set the learning rate for trajectory optimization.")
n_iters = st.sidebar.slider("🔁 Iterations", min_value=50, max_value=500, value=300, help="Set the number of iterations for optimization.")

# --- Multiple Obstacle Inputs ---
st.sidebar.header("🛑 Obstacles")
num_obstacles = st.sidebar.number_input("Number of Obstacles", min_value=0, max_value=10, value=2, step=1, help="Set the number of obstacles in the environment.")
obstacles = []
for i in range(num_obstacles):
    st.sidebar.subheader(f"🛑 Obstacle {i+1}")
    obstacle_x = st.sidebar.slider(f"X {i+1}", min_value=0.0, max_value=10.0, value=float(2 * i + 1), step=0.1, help=f"Set the X position of obstacle {i+1}.")
    obstacle_y = st.sidebar.slider(f"Y {i+1}", min_value=0.0, max_value=10.0, value=float(3 * i + 2), step=0.1, help=f"Set the Y position of obstacle {i+1}.")
    obstacle_radius = st.sidebar.slider(f"Radius {i+1}", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help=f"Set the radius of obstacle {i+1}.")
    obstacles.append((obstacle_x, obstacle_y, obstacle_radius))

# Color Inputs for Start and Goal
start_color = st.sidebar.color_picker("🌱 Start Color", "#00FF00", help="Pick a color for the start point.")
goal_color = st.sidebar.color_picker("🎯 Goal Color", "#FF0000", help="Pick a color for the goal point.")

# --- FAQ Section ---
with st.expander("💡 Frequently Asked Questions"):
    st.write("""
    **What is Model Predictive Control (MPC)?**  
    MPC is a control strategy that uses optimization to determine the best trajectory in a dynamic environment, ensuring smoothness and collision avoidance.
    
    **How does the tool work?**  
    The trajectory is optimized using a combination of AI and MPC algorithms that consider the start and goal positions, obstacles, and smoothness constraints.
    
    **What factors influence the optimization?**  
    - **Waypoints**: More waypoints result in a more detailed trajectory.
    - **Learning Rate**: Affects the optimization speed and accuracy.
    - **Obstacles**: More obstacles increase optimization complexity.
    """)

# --- Optimization Process ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start and Goal Points
start = torch.tensor([start_x, start_y], device=device, dtype=torch.float32)
goal = torch.tensor([goal_x, goal_y], device=device, dtype=torch.float32)

# Trajectory Variable (optimize all but the start point)
waypoints = torch.linspace(0, 1, n_waypoints).unsqueeze(1).to(device).float()
trajectory = start.unsqueeze(0) + (goal - start).unsqueeze(0) * waypoints + 0.5 * torch.randn(n_waypoints, 2, device=device).float()
trajectory = trajectory.detach().clone().requires_grad_(True)

# Optimizer
optimizer = torch.optim.Adam([trajectory], lr=lr)

# Loss Function (MPC)
def mpc_loss(traj, obstacles):
    loss = 0.0

    # Distance to goal
    loss += 5.0 * F.mse_loss(traj[-1], goal)

    # Smoothness (minimize differences between consecutive points)
    loss += 1.0 * torch.sum((traj[1:] - traj[:-1]) ** 2)

    # Obstacle penalty
    for ox, oy, r in obstacles:
        obstacle_tensor = torch.tensor([ox, oy], device=device, dtype=torch.float32)
        dist = torch.sqrt(torch.sum((traj - obstacle_tensor) ** 2, dim=1))
        collision = F.relu(r - dist)  # only penalize if inside
        loss += 10.0 * torch.sum(collision)

    return loss

# Optimization Loop with progress bar
progress_bar = st.progress(0)
for iter in range(n_iters):
    optimizer.zero_grad()
    traj = torch.cat([start.unsqueeze(0), trajectory[1:]], dim=0)  # fix start
    loss = mpc_loss(traj, obstacles)
    loss.backward()
    optimizer.step()

    if iter % 50 == 0:
        st.write(f"🛠️ Iteration {iter}: Loss = {loss.item():.4f}")
    
    # Update progress bar
    progress_bar.progress(int(((iter + 1) / n_iters) * 100))

# Plotting Function with Enhanced Styling
def plot_trajectory_with_obstacles(traj, obstacles, start=None, goal=None, start_color='#00FF00', goal_color='#FF0000'):
    xs, ys = traj[:, 0].cpu().numpy(), traj[:, 1].cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, '-o', label="Trajectory", color='blue', markersize=6, linestyle='-', linewidth=2)

    # Plot obstacles
    for ox, oy, r in obstacles:
        circle = plt.Circle((ox, oy), r, color='r', alpha=0.3, edgecolor='black', linewidth=1)
        plt.gca().add_patch(circle)

    # Mark start/goal with dynamic colors
    if start is not None:
        plt.scatter(start[0], start[1], c=start_color, s=150, label='Start', zorder=5, edgecolors='black', linewidth=2)
    if goal is not None:
        plt.scatter(goal[0], goal[1], c=goal_color, s=150, label='Goal', zorder=5, edgecolors='black', linewidth=2)

    # Additional plot formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Optimized Trajectory", fontsize=16, weight='bold')
    plt.legend(fontsize=12)
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Plot the optimized trajectory with obstacles
with torch.no_grad():
    final_traj = torch.cat([start.unsqueeze(0), trajectory[1:]], dim=0)
    plot_trajectory_with_obstacles(final_traj, obstacles, start=start.cpu().numpy(), goal=goal.cpu().numpy(), start_color=start_color, goal_color=goal_color)