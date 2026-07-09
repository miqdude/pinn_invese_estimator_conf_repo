import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class CoGDetectorPINN(nn.Module):
    def __init__(self):
        super(CoGDetectorPINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),          
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 2)    
        )
    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. MELLINGER SE(3) WRENCH ALLOCATOR
# ==========================================
class GeometricLeader:
    def __init__(self, use_pinn=False, weights_file_path="pinn_cog_weights.pth", mass_total = 8.0):
        self.M_total = mass_total # kg (Mass of rig + 4 drones + 1 Kg Payload)
        self.g = 9.81
        self.dt = 1.0 / 240.0 # PyBullet timestep
        
        # 1. Softer Position Control (Slow and steady)
        self.kp_pos = np.array([1.5, 1.5, 4.0])  # Dropped significantly
        self.kd_pos = np.array([2.0, 2.0, 3.0])  # Damping is now higher relative to P
        self.ki_pos = np.array([0.1, 0.1, 0.5])  # Lowered integral to prevent slow wind-up
        self.integral_pos_error = np.zeros(3) 
        
        # 2. Softer Attitude Control (Prioritizes stability over speed)
        self.kp_att = np.array([8.0, 8.0, 0.2])  # Dropped P gain 
        self.kd_att = np.array([4.0, 4.0, 0.1])  # Kept D high for heavy drag/damping this values gives slow oscilating
        # self.kp_att = np.array([0.8, .8, 0.2])  # Dropped P gain 
        # self.kd_att = np.array([0, 0, 0.1])  # Kept D high for heavy drag/damping this value works

        # 3. Geometric Mounting Positions
        self.r_arms = {
            'iris_1': np.array([ 0.5,  0.5, 0.05]), 
            'iris_2': np.array([ 0.5, -0.5, 0.05]), 
            'iris_3': np.array([-0.5,  0.5, 0.05]), 
            'iris_4': np.array([-0.5, -0.5, 0.05])  
        }
        
        self.c_yaw = [1.0, -1.0, -1.0, 1.0]
        
        # PINN Setup
        self.use_pinn = use_pinn
        # Make sure your CoGDetectorPINN class is imported/defined before this!
        # self.pinn = CoGDetectorPINN() 

        if self.use_pinn:
            try:
                weights = torch.load(weights_file_path, map_location=torch.device('cpu'), weights_only=True)
                self.pinn.load_state_dict(weights)
                self.pinn.eval()
                print("Successfully loaded CoG PINN weights!")
            except FileNotFoundError:
                print("WARNING: Weights not found! Using untrained PINN.")

    def get_allocation_matrix(self, cog_x=0.0, cog_y=0.0):
        """ 
        Builds the 4x4 Square Matrix.
        If the PINN provides a CoG offset, the lever arms adapt instantly!
        """
        A = np.zeros((4, 4))
        A[0, :] = 1.0 # Total Thrust 
        
        for i, (name, pos) in enumerate(self.r_arms.items()):
            # Shift the mathematical center based on PINN inference
            eff_x = pos[0] - cog_x
            eff_y = pos[1] - cog_y
            
            A[1, i] = eff_y           # Roll Moment 
            A[2, i] = -eff_x          # Pitch Moment 
            A[3, i] = self.c_yaw[i]   # Yaw Moment
            
        return A

    def get_follower_commands(self, dt, current_pos, current_vel, current_rpy, current_ang_vel, target_pos, target_vel, target_yaw):
        
        # ==========================================
        # Position Control (X, Y, Z)
        # ==========================================
        pos_error = np.array(target_pos) - np.array(current_pos)
        vel_error = np.array(target_vel) - np.array(current_vel)

        # Z-axis with Integrator
        self.integral_pos_error += pos_error[2] * dt
        self.integral_pos_error = np.clip(self.integral_pos_error, -1.5, 1.5) 

        # New calculate r_ddot_des
        # r_ddot_des_1 = self.g * ()
        
        # Calculate total thurst on Z
        F_B_des = self.M_total * (r_ddot_3 + self.g)
        
        # ==========================================
        # Attitude Control
        # ==========================================
        psi_T = current_rpy[2] # Current Yaw
        
        # Map the X/Y accelerations to desired Pitch and Roll 
        # (This is exactly the math from the paper screenshot!)
        phi_des = (1.0 / self.g) * (r_ddot_1 * np.sin(psi_T) - r_ddot_2 * np.cos(psi_T))
        theta_des = (1.0 / self.g) * (r_ddot_1 * np.cos(psi_T) + r_ddot_2 * np.sin(psi_T))
        
        # CRITICAL SAFETY CLAMP: 
        # Linear approximations only work for small angles. 
        # We clamp to 0.25 rad (~14 degrees) to ensure the frame stays stable during transit.
        phi_des = np.clip(phi_des, -0.25, 0.25)
        theta_des = np.clip(theta_des, -0.25, 0.25)
        
        # ==========================================
        # Calculate the desired acceleration (r_ddot_1 -> X, r_ddot_2 -> Y, r_ddot_3 -> Z)
        # ==========================================
        r_ddot_1 = (self.kp_pos[0] * pos_error[0]) + (self.kd_pos[0] * vel_error[0])
        r_ddot_2 = (self.kp_pos[1] * pos_error[1]) + (self.kd_pos[1] * vel_error[1])
        r_ddot_3 = (self.kp_pos[2] * pos_error[2]) + (self.kd_pos[2] * vel_error[2])
        # r_ddot_3 = (self.kp_pos[2] * pos_error[2]) + (self.kd_pos[2] * vel_error[2]) + (self.ki_pos[2] * self.integral_pos_error[2])
        
        att_error = np.array([phi_des - current_rpy[0], 
                              theta_des - current_rpy[1], 
                              target_yaw - current_rpy[2]])
        
        # 0-only hover attitude
        # att_error = np.array([0 - current_rpy[0], 
        #                       0 - current_rpy[1], 
        #                       target_yaw - current_rpy[2]])
                              
        rate_error = np.array([0.0, 0.0, 0.0]) - np.array(current_ang_vel)
        
        # Calculate stabilizing torques
        M_des = (self.kp_att * att_error) + (self.kd_att * rate_error)
        M_des[2] = np.clip(M_des[2], -2.0, 2.0) # Clamp yaw

        # ==========================================
        # Motor Allocation (With PINN placeholder)
        # ==========================================
        Wrench = np.array([F_B_des, M_des[0], M_des[1], M_des[2]])
        print(f"Wrench {M_des[0]}")

        # Build the dynamic matrix 
        A_nominal = self.get_allocation_matrix()
        
        # A_pinn = ... (This is where your PINN outputs will go later!)
        A_final = A_nominal # + A_pinn
        
        A_inv = np.linalg.pinv(A_final)
        
        # Solve for the exact thrust required by each of the 4 drones
        quad_thrusts = np.dot(A_inv, Wrench)
        
        follower_cmds = {}
        for i, name in enumerate(self.r_arms.keys()):
            follower_cmds[name] = max(0.0, quad_thrusts[i])
            
        return follower_cmds, Wrench, np.array([0.0, 0.0])