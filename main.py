#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pybullet as p
import pybullet_data
import socket
import json
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from geometric_control import GeometricControl
from mellinger_control import MellingerControl

from payload_env import PayloadEnv
from utils import get_trajectory, get_kinetic_energy, show_evaluation
from pinn_inverse_estimator import Inverse6DoFPINN, train_pinn_step

# --- PlotJuggler UDP Setup ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PLOTJUGGLER_ADDRESS = ('127.0.0.1', 9870)

# --- Constants ---
RATE_HZ = 240.0
TIME_STEP = 1.0 / RATE_HZ
GRAVITY_MSS = 9.81
SIMULATION_DURATION = 20 # simulation time 3 minutes
MAX_DIST_ERR = 1.5
TOTAL_EPISODES = 1000
MAX_DRONE_THRUST = 50
TOTAL_SYSTEM_MASS = 7.8

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
parser = argparse.ArgumentParser(description="PINN simulation of Quadrotors lifting a pendulum weight")

parser.add_argument("-f", type=str, default="final_assembly.urdf", help="file path to the environment")
parser.add_argument("-fl", type=str, default="virtual_leader.urdf", help="urdf file to the leader")
parser.add_argument("-i", type=bool, default=False, help="use AI inference")
parser.add_argument("-w", type=str, default="pinn_", help="file containing the weights if using AI inference")
parser.add_argument("-r", type=str, help="flight data in csv")
parser.add_argument("--debug", type=bool, default=False, help="use visual debug in simulation")
parser.add_argument("--vis", type=bool, default=False, help="show simulation")

args = parser.parse_args()

# --- PyBullet initialization ---
if args.vis:
    p.connect(p.GUI) # showing visualization
else:
    p.connect(p.DIRECT)

p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, -GRAVITY_MSS)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

leader = MellingerControl(use_pinn=args.i, weights_file_path=args.w, mass_total= TOTAL_SYSTEM_MASS)
# leader = GeometricControl(use_pinn=args.i, weights_file_path=args.w, mass_total= TOTAL_SYSTEM_MASS)

if args.i:
    print(f"Using INFERENCE MODE")

env = PayloadEnv(args.f, args.fl, leader, p, GRAVITY_MSS, debug=args.debug, drone_max_thrust=MAX_DRONE_THRUST)
true_cog = env.reset()
time_now = 0.0

target_position = (0.0, 0.0, 4.0)
target_yaw = 0.0

# ... (Environment initialization and trajectory functions stay here) ...
print("="*50)
print(f"Simulation Environment running, duration {SIMULATION_DURATION/60} mins.")
print("Streaming to PlotJuggler on port 9870...")
print("Press Ctrl+C in the terminal to stop")

if args.r:
    print("RECORDING DATA to CSV")

# 1. Initialize the empty dataset list and define the column headers
flight_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}...")

pinn_model = Inverse6DoFPINN().to(device)

# 2. Load the saved dictionary into the model
pinn_model.load_state_dict(torch.load("calibration_results/inverse_pinn_6dof_final.pth"))

# 3. CRITICAL: Set to evaluation mode if you are just running inference!
# This disables dropout and batch normalization layers if you added any later.
pinn_model.eval() 

print("Successfully loaded PINN weights from disk.")

# Masses in kg
mass_frame = 1.5 * 4 + 0.8 # drone mass + frame
mass_payload = 1.0

# Payload offset [x, y, z] in meters (e.g., shifted 20cm forward and 20cm right)
offset = np.array(env.reset()) # get the payload offset

# Base inertia matrix of the empty drone frame
J_base = torch.diag(torch.tensor([1.515, 1.515, 3.000], dtype=torch.float32, device=device))

try:
    curr_episode = 0
    while curr_episode < TOTAL_EPISODES:
        
        print("*"*36)
        print(f"{curr_episode}-th EPISODE IS STARTING")
        true_cog = env.reset()
        print(f"New CoG Anomaly Generated at: X={true_cog[0]:.3f}, Y={true_cog[1]:.3f}")

        # 3. Reset the simulation clock
        time_now = 0.0

        while time_now <= SIMULATION_DURATION:
            # Get the current dynamic target (Circle or Figure-8)
            target_pos, target_vel, target_acc, target_rpy, target_ang_vel = get_trajectory(time_now, path_type="hover")
            target_yaw = 0.0 

            # Step the Multi-Agent Environment
            pos, current_rpy, current_vel, current_ang_vel, all_thrusts, wrenching_force, cog_true, cog_guess, phi_des, theta_des, quad_thrust = env.step(
                TIME_STEP,
                time_now,
                target_pos,
                target_vel,
                target_ang_vel,
                target_acc,
                target_yaw
            )

            # Calculate the 3D Tracking Error
            pos_error = np.array(pos) - np.array(target_pos)
            distance_err = np.linalg.norm(pos_error)

            # calculate loss

            #   Append the 14 variables to our dataset for this specific timestep
            row = {
                "episode_num": curr_episode,
                "time_t": time_now,
                "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                "roll": current_rpy[0], "pitch": current_rpy[1], "yaw": current_rpy[2],
                "velocity_x": current_vel[0], "velocity_y": current_vel[1], "velocity_z": current_vel[2],
                "ang_vel_x": current_ang_vel[0], "ang_vel_y": current_ang_vel[1], "ang_vel_z": current_ang_vel[2],
                "target_pos_x": target_pos[0], "target_pos_y": target_pos[1], "target_pos_z": target_pos[2],
                "target_yaw": target_yaw,
                "target_vel_x": target_vel[0], "target_vel_y": target_vel[1], "target_vel_z": target_vel[2],
                "target_ang_vel_x": target_ang_vel[0], "target_ang_vel_y": target_ang_vel[1], "target_ang_vel_z": target_ang_vel[2],
                "target_acc_x": target_acc[0], "target_acc_y": target_acc[1], "target_acc_z": target_acc[2],
                "thrust_1": quad_thrust[0], "thrust_2": quad_thrust[1], "thrust_3": quad_thrust[2], "thrust_4": quad_thrust[3],
                "wrench_thrust": wrenching_force[0], "wrench_roll": wrenching_force[1], "wrench_pitch": wrenching_force[2], "wrench_yaw": wrenching_force[3],
                "guess_cog_x": cog_guess[0], "guess_cog_y": cog_guess[1],
                "true_cog_x": cog_true[0], "true_cog_y": cog_true[1],
            }
            flight_data.append(row)
            
            # ==========================================
            # CRASH DETECTION & AUTO-RESET
            # ==========================================
            if abs(current_rpy[0]) > 1.5 or abs(current_rpy[1]) > 1.5 or distance_err > MAX_DIST_ERR :
                
                print(f"Episode Failed! Drifted {distance_err:.2f}m off target. Resetting...")
                print(f"CRASH DETECTED at t={time_now:.2f}s! Resetting environment...") 
                
                # Pause for a second so you can see the reset happen
                time.sleep(2.0) 
                break # Skip the rest of this loop and start fresh

            telemetry = {
                "timestamp": time_now,
                "target": {
                    "pos_x":    target_pos[0],
                    "pos_y":    target_pos[1],
                    "pos_z":    target_pos[2],
                    "vel_x":    target_vel[0],
                    "vel_y":    target_vel[1],
                    "vel_z":    target_vel[2],
                    "ang_z":    target_yaw, 
                },
                "actual_frame": {
                    "pos_x":    pos[0],
                    "pos_y":    pos[1],
                    "pos_z":    pos[2],
                    "roll":     current_rpy[0],
                    "pitch":    current_rpy[1],
                    "yaw":      current_rpy[2],
                    "vel_x":    current_vel[0],
                    "vel_y":    current_vel[1],
                    "vel_z":    current_vel[2],
                    "ang_z":    current_ang_vel[2],
                    "phi_des": phi_des,
                    "theta_des": theta_des
                },
                "drone_thrusts": { f"drone{i}": float(all_thrusts[i]) for i in range(len(all_thrusts)) },
                "CoG" : {
                    "true_cog_x": cog_true[0],
                    "true_cog_y": cog_true[1],
                    "guessed_cog_x": cog_guess[0],
                    "guessed_cog_y": cog_guess[1]
                }
            }
            
            sock.sendto(json.dumps(telemetry).encode('utf-8'), PLOTJUGGLER_ADDRESS)

            time_now += TIME_STEP
            time.sleep(TIME_STEP)

    
        print(f"{curr_episode}-th EPISODE has ended with total time {time_now} seconds")
        curr_episode += 1

    print("\n--- Simulation has ended ---")
    print(f"Collected {len(flight_data)} frames of flight data.")

    # show_evaluation(env, time_now) # need to update this evaluation
    
    # Convert the raw list into a Pandas DataFrame and save to CSV
    if args.r:
        df = pd.DataFrame(flight_data)
        df.to_csv(args.r, index=False)
    
        print(f"Successfully saved dataset to {args.r}!")
    else:
        print("No dataset saved!")

# 3. Graceful Exit and File Saving
except KeyboardInterrupt:
    
    print("\n--- Simulation stopped by user---")
    
    # Convert the raw list into a Pandas DataFrame and save to CSV
    if args.r:
        df = pd.DataFrame(flight_data)
        df.to_csv(args.r, index=False)
    
        print(f"Successfully saved dataset to {args.r}!")
    else:
        print("No dataset saved!")

    # Disconnect PyBullet safely
    p.disconnect()