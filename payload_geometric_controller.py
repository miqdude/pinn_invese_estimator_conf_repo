import numpy as np

class PayloadGeometricController:
    def __init__(self, mass_payload=2.5):
        self.m = mass_payload
        self.g = 9.81

        self.kp_pos = np.array([3.0, 3.0, 3.0]) 
        self.ka_ff  = np.array([0.15, 0.15, 0.0]) 
        self.kp_att = np.array([0.0, 0.0, 1.0])  

        # 1. Split the compensation into Proportional (Spring) and Derivative (Damper)
        self.k_comp_p = 0.005  # The base counter-steer
        self.k_comp_d = 0.002  # THE SHOCK ABSORBER: Fights the swing velocity
        
        self.integral_z = 0.0
        
        # 2. Memory to calculate angular velocity without changing main.py
        self.prev_theta = 0.0
        self.prev_phi = 0.0

        self.r_arms = {
            'iris_1': np.array([0.5, 0.5, 0.0]),
            'iris_2': np.array([-0.5, 0.5, 0.0]),
            'iris_3': np.array([-0.5, -0.5, 0.0]),
            'iris_4': np.array([0.5, -0.5, 0.0])
        }

    def get_follower_velocities(self, dt, current_pos, current_rpy, target_pos, target_vel, target_accel, target_yaw, theta_roll, phi_pitch):
        
        # Calculate Positional Error with Deadband
        pos_error = np.array(target_pos) - np.array(current_pos)
        if abs(pos_error[0]) < 0.015: pos_error[0] = 0.0
        if abs(pos_error[1]) < 0.015: pos_error[1] = 0.0
        
        # Z-Axis Integral (Perfect Hover)
        self.integral_z += pos_error[2] * dt
        self.integral_z = np.clip(self.integral_z, -2.0, 2.0) 

        # Baseline Tracking 
        v_base = (np.array(target_vel) + 
                  (self.kp_pos * pos_error) + 
                  (self.ka_ff * np.array(target_accel)))
        v_base[2] += (1.5 * self.integral_z)

        # --- THE FIX: PENDULUM DERIVATIVE DAMPING ---
        # 1. Calculate how fast the pendulum is swinging (rad/s)
        theta_dot = (theta_roll - self.prev_theta) / dt
        phi_dot = (phi_pitch - self.prev_phi) / dt
        
        # Save current angles for the next millisecond
        self.prev_theta = theta_roll
        self.prev_phi = phi_pitch

        # 2. Add the Damping term to the Euler-Lagrange Force
        # Notice we are multiplying the velocity by our new Kd gain
        f_dist_x = self.m * self.g * (np.sin(phi_pitch) + (self.k_comp_d * phi_dot))
        f_dist_y = self.m * self.g * (np.sin(theta_roll) + (self.k_comp_d * theta_dot))
        f_disturbance = np.array([f_dist_x, f_dist_y, 0.0])

        # Apply compensation (Assuming the '+' sign from your last test is correct)
        v_compensation = self.k_comp_p * f_disturbance
        v_leader_raw = v_base + v_compensation

        v_leader = np.clip(v_leader_raw, -5.0, 5.0)

        # Attitude & Rigid Frame Distribution
        att_error = np.array([0.0, 0.0, target_yaw - current_rpy[2]])
        omega_leader = self.kp_att * att_error

        follower_cmds = {}
        for drone_name, r_i in self.r_arms.items():
            v_i = v_leader + np.cross(omega_leader, r_i)
            v_i = np.clip(v_i, -7.0, 7.0)
            follower_cmds[drone_name] = v_i

        return follower_cmds