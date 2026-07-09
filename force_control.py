import numpy as np
import time

class PIDController:
    """Standard PID controller with anti-windup."""
    def __init__(self, kp, ki, kd, limits=(None, None), is_angle=False):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.min_out, self.max_out = limits
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.is_angle = is_angle

    def update(self, setpoint, measured_value):
        now = time.time()
        dt = max(now - self.prev_time, 1e-4)
        
        error = setpoint - measured_value
        # --- NEW: Shortest path angle wrapping ---
        if self.is_angle:
            error = (error + np.pi) % (2 * np.pi) - np.pi

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        if self.min_out is not None: output = max(self.min_out, output)
        if self.max_out is not None: output = min(self.max_out, output)
        
        self.prev_error = error
        self.prev_time = now
        return output

class ForceController:
    """Cascaded controller: Velocity -> Attitude -> Force (Newtons)"""
    def __init__(self, mass=0.27, g=9.81):
        self.mass = mass
        self.g = g
        
        # 1. OUTER LOOP: Velocity Controllers (Output: Desired Acceleration in m/s^2)
        self.pid_vx = PIDController(kp=2.0, ki=0.1, kd=0.5, limits=(-5, 5))
        self.pid_vy = PIDController(kp=2.0, ki=0.1, kd=0.5, limits=(-5, 5))
        self.pid_vz = PIDController(kp=3.0, ki=0.5, kd=1.0, limits=(-5, 5))

        # Aggressive control
        # self.pid_vx = PIDController(kp=4.0, ki=0.1, kd=1.0, limits=(-10.0, 10.0))
        # self.pid_vy = PIDController(kp=4.0, ki=0.1, kd=1.0, limits=(-10.0, 10.0))
        # self.pid_vz = PIDController(kp=5.0, ki=0.5, kd=1.5, limits=(-10.0, 10.0))
        
        # 2. INNER LOOP: Attitude Controllers (Output: Differential Thrust in Newtons)
        # Limits and gains are scaled down for a 0.027 kg drone. 
        # Total hover thrust is ~0.26N (0.066N per motor).
        self.pid_roll = PIDController(kp=0.5, ki=0.0, kd=0.15, limits=(-0.1, 0.1), is_angle=True)
        self.pid_pitch = PIDController(kp=0.5, ki=0.0, kd=0.15, limits=(-0.1, 0.1), is_angle=True)

        self.pid_yaw_rate = PIDController(kp=0.05, ki=0.01, kd=0.0, limits=(-0.1, 0.1), is_angle=False)

        # Aggressive control
        # self.pid_roll = PIDController(kp=6.0, ki=0.0, kd=1.5, limits=(-10.0, 10.0), is_angle=True)
        # self.pid_pitch = PIDController(kp=6.0, ki=0.0, kd=1.5, limits=(-10.0, 10.0), is_angle=True)

        # self.pid_yaw_rate = PIDController(kp=1.0, ki=0.1, kd=0.0, limits=(-4.0, 4.0), is_angle=False)

    def get_motor_thrusts(self, state, target_velocities, target_angular):
        """
        state: dict containing 'v' (vx, vy, vz), 'rpy' (roll, pitch, yaw)
        target_velocities: tuple (vx_des, vy_des, vz_des)
        Returns: A list of thrust outputs for each motor directly in Newtons [f1, f2, f3, f4]
        """
        v_current = state['v'] # velocity
        w_current = state['w'] # angular
        rpy_current = state['rpy']
        current_yaw = rpy_current[2]

        # --- STEP 1: OUTER LOOP (Get desired accelerations) ---
        a_x_des = self.pid_vx.update(target_velocities[0], v_current[0])
        a_y_des = self.pid_vy.update(target_velocities[1], v_current[1])
        a_z_des = self.pid_vz.update(target_velocities[2], v_current[2])

        # --- STEP 2: PHYSICS MAPPING (Accel to Angles) ---
        pitch_des = (a_x_des * np.cos(current_yaw) + a_y_des * np.sin(current_yaw)) / self.g
        roll_des  = (a_x_des * np.sin(current_yaw) - a_y_des * np.cos(current_yaw)) / self.g
        
        max_angle = np.radians(15) # degree to radians
        pitch_des = np.clip(pitch_des, -max_angle, max_angle)
        roll_des = np.clip(roll_des, -max_angle, max_angle)

        # --- STEP 3: INNER LOOP (Get Differential Force Corrections) ---
        # These outputs now represent the extra Newtons of force to apply/subtract for rotation
        roll_corr  = self.pid_roll.update(roll_des, rpy_current[0])
        pitch_corr = self.pid_pitch.update(pitch_des, rpy_current[1])
        
        yaw_corr   = self.pid_yaw_rate.update(target_angular[2], w_current[2])
        
        # --- STEP 4: BASE Z-AXIS FORCE ---
        # F = m * (g + a). We divide by 4 to get the baseline force needed per motor.
        total_z_force = self.mass * (self.g + a_z_des)
        thrust_base = total_z_force / 4.0

        # --- STEP 5: MOTOR MIXING (Forces in Newtons) ---
        # f1 = thrust_base - roll_corr - pitch_corr + yaw_corr
        # f2 = thrust_base + roll_corr + pitch_corr + yaw_corr
        # f3 = thrust_base + roll_corr - pitch_corr - yaw_corr
        # f4 = thrust_base - roll_corr + pitch_corr - yaw_corr

        # Try reversing the yaw_corr signs like this:
        f1 = thrust_base - roll_corr - pitch_corr - yaw_corr # Changed from + to -
        f2 = thrust_base + roll_corr + pitch_corr - yaw_corr # Changed from + to -
        f3 = thrust_base + roll_corr - pitch_corr + yaw_corr # Changed from - to +
        f4 = thrust_base - roll_corr + pitch_corr + yaw_corr # Changed from - to +

        # Return thrusts, ensuring a motor never asks for negative force
        return [max(0.0, f1), max(0.0, f2), max(0.0, f3), max(0.0, f4)]