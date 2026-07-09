import numpy as np
import time

class PIDController:
    """Standard PID controller with anti-windup."""
    def __init__(self, kp, ki, kd, limits=(None, None)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.min_out, self.max_out = limits
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, setpoint, measured_value):
        now = time.time()
        dt = max(now - self.prev_time, 1e-4)
        
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        if self.min_out is not None: output = max(self.min_out, output)
        if self.max_out is not None: output = min(self.max_out, output)
        
        self.prev_error = error
        self.prev_time = now
        return output

class VelocityController:
    """Cascaded controller: Velocity -> Attitude -> RPM"""
    def __init__(self, mass=0.027, g=9.81, hover_rpm=1000):
        self.mass = mass
        self.g = g
        self.hover_rpm = hover_rpm
        
        # 1. OUTER LOOP: Velocity Controllers (Output: Desired Acceleration)
        self.pid_vx = PIDController(kp=2.0, ki=0.1, kd=0.5, limits=(-5, 5))
        self.pid_vy = PIDController(kp=2.0, ki=0.1, kd=0.5, limits=(-5, 5))
        self.pid_vz = PIDController(kp=3.0, ki=0.5, kd=1.0, limits=(-5, 5))
        
        # 2. INNER LOOP: Attitude Controllers (Output: Desired Torque/Correction)
        # Limits here represent the maximum RPM correction applied for tilting
        self.pid_roll = PIDController(kp=50.0, ki=0.0, kd=15.0, limits=(-500, 500))
        self.pid_pitch = PIDController(kp=50.0, ki=0.0, kd=15.0, limits=(-500, 500))
        self.pid_yaw = PIDController(kp=30.0, ki=0.0, kd=5.0, limits=(-200, 200))

    def get_motor_rpms(self, state, target_velocities, target_yaw):
        """
        state: dict containing 'v' (vx, vy, vz), 'rpy' (roll, pitch, yaw)
        target_velocities: tuple (vx_des, vy_des, vz_des)
        """
        v_current = state['v']
        rpy_current = state['rpy']
        current_yaw = rpy_current[2]

        # --- STEP 1: OUTER LOOP (Get desired accelerations) ---
        a_x_des = self.pid_vx.update(target_velocities[0], v_current[0])
        a_y_des = self.pid_vy.update(target_velocities[1], v_current[1])
        a_z_des = self.pid_vz.update(target_velocities[2], v_current[2])

        # --- STEP 2: PHYSICS MAPPING (Accel to Angles) ---
        # Calculate target roll and pitch based on desired X/Y acceleration and current yaw
        pitch_des = (a_x_des * np.cos(current_yaw) + a_y_des * np.sin(current_yaw)) / self.g
        roll_des  = (a_x_des * np.sin(current_yaw) - a_y_des * np.cos(current_yaw)) / self.g
        
        # Clip desired angles to prevent the drone from flipping over (e.g., max 30 degrees)
        max_angle = np.radians(30)
        pitch_des = np.clip(pitch_des, -max_angle, max_angle)
        roll_des = np.clip(roll_des, -max_angle, max_angle)

        # --- STEP 3: INNER LOOP (Get Torque Corrections) ---
        roll_corr  = self.pid_roll.update(roll_des, rpy_current[0])
        pitch_corr = self.pid_pitch.update(pitch_des, rpy_current[1])
        yaw_corr   = self.pid_yaw.update(target_yaw, current_yaw)
        
        # Thrust correction (Z-axis)
        # We add hover_rpm as the baseline required to counteract gravity
        thrust_base = a_z_des * 1000

        # --- STEP 4: MOTOR MIXING ---
        m1 = thrust_base - roll_corr - pitch_corr + yaw_corr
        m2 = thrust_base + roll_corr + pitch_corr + yaw_corr
        m3 = thrust_base + roll_corr - pitch_corr - yaw_corr
        m4 = thrust_base - roll_corr + pitch_corr - yaw_corr

        return [max(0, m1), max(0, m2), max(0, m3), max(0, m4)]