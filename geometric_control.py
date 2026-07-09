import numpy as np

class GeometricControl:
    def __init__(self, use_pinn=False, weights_file_path="pinn_cog_weights.pth", mass_total=8.0):
        self.M_total = mass_total # kg (Mass of rig + 4 drones + 1 Kg Payload)
        self.g = 9.81
        self.dt = 1.0 / 240.0 # PyBullet timestep
        
        # Position Control 
        self.kp_pos = np.array([1.0, 1.0, 1.5])  
        self.kd_pos = np.array([1.5, 1.5, 2.0])  
        self.ki_pos = np.array([0.0, 0.0, 0.0])  
        self.integral_pos_error = np.zeros(3) 
        
        # Attitude Control (Lowered gains to prevent high-frequency oscillations)
        self.kp_att = np.array([4.0, 4.0, 0.3])  
        self.kd_att = np.array([2.0, 2.0, 0.1])  

        # Geometric Mounting Positions
        self.r_arms = {
            'iris_1': np.array([ 0.5,  0.5, 0.05]), 
            'iris_2': np.array([ 0.5, -0.5, 0.05]), 
            'iris_3': np.array([-0.5,  0.5, 0.05]), 
            'iris_4': np.array([-0.5, -0.5, 0.05])  
        }

    def get_allocation_matrix(self, cog_x=0.0, cog_y=0.0):
        """ 
        Builds a decoupled 3x4 Matrix (Total Thrust, Roll, Pitch). 
        Yaw is intentionally excluded from vertical thrust allocation.
        """
        A = np.zeros((3, 4))
        A[0, :] = 1.0 
        
        for i, (name, pos) in enumerate(self.r_arms.items()):
            eff_x = pos[0] - cog_x
            eff_y = pos[1] - cog_y
            
            A[1, i] = eff_y           # Roll Moment (Y arm)
            A[2, i] = -eff_x          # Pitch Moment (-X arm)
            
        return A
    
    def _calculate_total_cog(self, m_frame, m_payload, payload_offset):
        """ Internal helper to find the true physical center of rotation """
        m_tot = m_frame + m_payload
        if m_tot <= 0.0:
            return 0.0, 0.0
            
        cog_x = (m_payload * payload_offset[0]) / m_tot
        cog_y = (m_payload * payload_offset[1]) / m_tot
        return cog_x, cog_y

    def get_follower_commands(
            self,
            payload_offset, dt, time_now,
            current_pos, current_rpy, current_vel, current_ang_vel, current_acc, current_ang_acc,
            target_pos, target_vel, target_ang_vel, target_acc, target_yaw,
            payload_offset_x, payload_offset_y, m_frame, m_payload,
        ):
        
        # ==========================================
        # 1. Position Control
        # ==========================================
        err_pos = np.array(target_pos) - np.array(current_pos)
        err_vel = np.array(target_vel) - np.array(current_vel)
        
        self.integral_pos_error += err_pos * dt
        
        r_ddot_des = (self.kp_pos * err_pos) + (self.kd_pos * err_vel) + (self.ki_pos * self.integral_pos_error) + target_acc
        
        # Total desired force vector
        F_des = self.M_total * (r_ddot_des + np.array([0.0, 0.0, self.g]))
        F_B_des = np.linalg.norm(F_des)

        # ==========================================
        # 2. SE(3) Geometric Attitude Controller (Yaw Disabled)
        # ==========================================
        z_des = F_des / (F_B_des + 1e-6) 
        
        # CONTINUOUS ALIGNMENT: Force target yaw to match current yaw
        # This prevents the matrix solver from commanding impossible twists
        yaw_cmd = current_rpy[2] 
            
        x_c = np.array([np.cos(yaw_cmd), np.sin(yaw_cmd), 0.0])
        
        # Construct full Desired Rotation Matrix
        y_des = np.cross(z_des, x_c)
        y_des = y_des / (np.linalg.norm(y_des) + 1e-6)
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack((x_des, y_des, z_des))
        
        # Construct Current Rotation Matrix (From RPY)
        cr, sr = np.cos(current_rpy[0]), np.sin(current_rpy[0])
        cp, sp = np.cos(current_rpy[1]), np.sin(current_rpy[1])
        cy, sy = np.cos(current_rpy[2]), np.sin(current_rpy[2])
        
        R_curr = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        
        # Geometric Error Vectors mapped via the Vee operator
        err_matrix = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        e_R = np.array([err_matrix[2, 1], err_matrix[0, 2], err_matrix[1, 0]])
        
        # BODY FRAME TRANSFORMATION: Crucial for preventing cross-coupled spirals
        omega_curr_body = R_curr.T @ np.array(current_ang_vel)
        omega_targ_body = R_curr.T @ np.array(target_ang_vel)
        e_omega = omega_curr_body - omega_targ_body
        
        # Calculate resulting M_des 
        M_des = -(self.kp_att * e_R) - (self.kd_att * e_omega)
        
        # NEUTRALIZE YAW: Hardcode to zero to respect actuator saturation limits
        M_des[2] = 0.0 
        
        F_B_cmd = np.dot(F_des, R_curr[:, 2])

        # ==========================================
        # 3. Classical Dynamics Allocation (Decoupled)
        # ==========================================
        true_cog_x, true_cog_y = self._calculate_total_cog(m_frame, m_payload, [payload_offset_x, payload_offset_y])

        # 3x4 Matrix: Only Thrust, Roll, and Pitch
        A_final = self.get_allocation_matrix(true_cog_x, true_cog_y)
        A_inv = np.linalg.pinv(A_final)
        
        Wrench_3D = np.array([max(0.0, F_B_cmd), M_des[0], M_des[1]])
        quad_thrusts = np.dot(A_inv, Wrench_3D)

        # ---------------------------------------------------------
        # COMMAND PACKAGING
        # ---------------------------------------------------------
        follower_thrust_cmds = {}
        follower_torque_cmds = {}
        
        # Drones receive 0.0 local yaw torque, relying entirely on natural drag
        yaw_torque_per_drone = 0.0 
        
        for i, name in enumerate(self.r_arms.keys()):
            follower_thrust_cmds[name] = max(0.0, quad_thrusts[i])
            follower_torque_cmds[name] = np.array([0.0, 0.0, yaw_torque_per_drone])
            
        Wrench_Full = np.array([F_B_cmd, M_des[0], M_des[1], M_des[2]])
        
        return follower_thrust_cmds, follower_torque_cmds, Wrench_Full