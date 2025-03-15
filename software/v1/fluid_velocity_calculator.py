import numpy as np

def exit_velocity(pressure):
    """Calculate water exit velocity using Torricelli's theorem."""
    pressure = pressure * 6894.76
    rho = 1000  # kg/m³ (density of water)
    return np.sqrt(2 * pressure / rho)

def find_launch_angle(v0, h0, z_target):
    """Find the shallower launch angle for the stream to hit the target depth."""
    g = 9.81  # Gravity (m/s^2)
    
    # Function to calculate time to reach the target depth (z_target)
    def time_to_hit_target(theta):
        """Solve for time given the launch angle."""
        theta_rad = np.radians(theta)
        v0y = v0 * np.sin(theta_rad)
        discriminant = v0y**2 + 2 * g * (h0 - z_target)
        if discriminant < 0:
            return None  # No valid time
        t = (v0y - np.sqrt(discriminant)) / g  # Use negative root for shallower trajectory
        return t

    # Function to calculate horizontal distance at a given time
    def horizontal_distance(theta, t):
        """Calculate horizontal distance given time and angle."""
        return v0 * np.cos(np.radians(theta)) * t

    # Try different angles and find the shallower one
    angles = np.linspace(1, 89, 1000)  # Sweep from 1 to 89 degrees
    best_angle = None
    min_diff = float('inf')

    for angle in angles:
        t = time_to_hit_target(angle)
        if t is None:
            continue  # Skip invalid times
        
        x = horizontal_distance(angle, t)
        diff = abs(x - z_target)  # Difference between actual and target horizontal distance
        
        if diff < min_diff:
            min_diff = diff
            best_angle = angle
    
    return best_angle + 45
