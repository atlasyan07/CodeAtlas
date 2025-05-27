#!/usr/bin/env python3
import numpy as np
import json
import argparse
import datetime
from pathlib import Path

def generate_telemetry(duration_sec=300, sampling_rate_hz=1, output_file="spacecraft_telemetry.json"):
    """
    Generate simulated spacecraft telemetry data and save it to a JSON file.
    
    Parameters:
    -----------
    duration_sec : int
        Duration of the simulation in seconds (default: 300s = 5 minutes)
    sampling_rate_hz : int
        Sampling rate in Hz (default: 1 Hz)
    output_file : str
        Output JSON file path
    """
    # Create time array
    num_samples = int(duration_sec * sampling_rate_hz)
    time = np.linspace(0, duration_sec, num_samples)
    
    # --- Quaternion Rotation (slew simulation) ---
    def simulate_quaternion_rotation(t):
        # Create a more interesting attitude profile with multiple maneuvers
        # Normalized time (0 to 1)
        t_norm = t / duration_sec
        
        # Base quaternion (identity rotation)
        qw = np.ones_like(t)
        qx = np.zeros_like(t)
        qy = np.zeros_like(t)
        qz = np.zeros_like(t)
        
        # Add some sinusoidal variations around multiple axes
        angle1 = np.pi/4 * np.sin(2 * np.pi * t_norm * 2)  # Slower oscillation
        angle2 = np.pi/6 * np.sin(2 * np.pi * t_norm * 5)  # Faster oscillation
        angle3 = np.pi/8 * np.cos(2 * np.pi * t_norm * 3)  # Medium oscillation
        
        # Calculate quaternion components for each time step
        for i in range(len(t)):
            # Create component quaternions
            q1 = np.array([np.cos(angle1[i]/2), np.sin(angle1[i]/2), 0, 0])  # X-axis rotation
            q2 = np.array([np.cos(angle2[i]/2), 0, np.sin(angle2[i]/2), 0])  # Y-axis rotation
            q3 = np.array([np.cos(angle3[i]/2), 0, 0, np.sin(angle3[i]/2)])  # Z-axis rotation
            
            # Multiply quaternions (q1 * q2 * q3)
            # This is a simplified quaternion multiplication
            qw[i] = q1[0]*q2[0]*q3[0] - q1[1]*q2[1]*q3[0] - q1[0]*q2[2]*q3[1] - q1[0]*q2[0]*q3[2]
            qx[i] = q1[0]*q2[0]*q3[1] + q1[0]*q2[1]*q3[0] + q1[1]*q2[0]*q3[0] - q1[0]*q2[2]*q3[2]
            qy[i] = q1[0]*q2[0]*q3[2] + q1[0]*q2[2]*q3[0] + q1[2]*q2[0]*q3[0] - q1[1]*q2[0]*q3[1]
            qz[i] = q1[0]*q2[1]*q3[2] + q1[0]*q2[2]*q3[1] + q1[1]*q2[0]*q3[2] - q1[1]*q2[2]*q3[0]
        
        # Normalize quaternions
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
        
        return qw, qx, qy, qz
    
    qw, qx, qy, qz = simulate_quaternion_rotation(time)
    
    # --- Body Rates (rad/s) ---
    # More realistic body rates that are consistent with the quaternion changes
    body_rate_x = 0.02 * np.sin(0.1 * time) + 0.01 * np.cos(0.05 * time)
    body_rate_y = 0.015 * np.cos(0.05 * time) + 0.008 * np.sin(0.08 * time)
    body_rate_z = 0.01 * np.sin(0.07 * time) + 0.005 * np.cos(0.12 * time)
    
    # --- Spacecraft Momentum (mNms) ---
    momentum_x = 10 + 2 * np.sin(0.02 * time)
    momentum_y = 15 + 3 * np.cos(0.015 * time)
    momentum_z = 12 + 1.5 * np.sin(0.025 * time)
    
    # --- Sun Vector in Body Frame ---
    # Create a more realistic sun vector that moves slowly across the field of view
    sun_theta = 0.005 * time  # very slow rotation
    sun_phi = 0.003 * time + np.pi/4  # elevation angle changing
    
    sun_vector_x = np.cos(sun_theta) * np.cos(sun_phi)
    sun_vector_y = np.sin(sun_theta) * np.cos(sun_phi)
    sun_vector_z = np.sin(sun_phi)
    
    # Normalize to unit vectors
    norm = np.sqrt(sun_vector_x**2 + sun_vector_y**2 + sun_vector_z**2)
    sun_vector_x /= norm
    sun_vector_y /= norm
    sun_vector_z /= norm
    
    # --- Nadir Vector (pointing to Earth) ---
    # Create a nadir vector that's roughly in the opposite direction of the sun
    # but with some variation to simulate orbital motion
    nadir_theta = sun_theta + np.pi + 0.1 * np.sin(0.01 * time)
    nadir_phi = -sun_phi + 0.1 * np.cos(0.01 * time)
    
    nadir_vector_x = np.cos(nadir_theta) * np.cos(nadir_phi)
    nadir_vector_y = np.sin(nadir_theta) * np.cos(nadir_phi)
    nadir_vector_z = np.sin(nadir_phi)
    
    # Normalize
    norm = np.sqrt(nadir_vector_x**2 + nadir_vector_y**2 + nadir_vector_z**2)
    nadir_vector_x /= norm
    nadir_vector_y /= norm
    nadir_vector_z /= norm
    
    # --- Velocity Vector (perpendicular to nadir in the orbital plane) ---
    velocity_vector_x = -nadir_vector_y
    velocity_vector_y = nadir_vector_x
    velocity_vector_z = np.zeros_like(nadir_vector_z)
    
    # Normalize
    norm = np.sqrt(velocity_vector_x**2 + velocity_vector_y**2 + velocity_vector_z**2)
    velocity_vector_x /= norm
    velocity_vector_y /= norm
    velocity_vector_z /= norm
    
    # Create a simulation sequence for the OrbitViz JSON format
    simulation_sequence = []
    
    # Current date and time
    current_time = datetime.datetime.now()
    
    for i in range(len(time)):
        sequence_entry = {
            "time": int(time[i]),
            "quaternion": {
                "w": float(qw[i]),
                "x": float(qx[i]),
                "y": float(qy[i]),
                "z": float(qz[i])
            },
            "bodyRates": {
                "x": float(body_rate_x[i]),
                "y": float(body_rate_y[i]),
                "z": float(body_rate_z[i])
            }
        }
        simulation_sequence.append(sequence_entry)
    
    # Construct the JSON data structure
    telemetry_data = {
        "metadata": {
            "name": "Simulated Spacecraft Attitude Data",
            "description": f"Generated telemetry data for {duration_sec} seconds at {sampling_rate_hz} Hz",
            "created": current_time.isoformat()
        },
        "attitude": {
            "w": float(qw[0]),
            "x": float(qx[0]),
            "y": float(qy[0]),
            "z": float(qz[0])
        },
        "bodyRates": {
            "x": float(body_rate_x[0]),
            "y": float(body_rate_y[0]),
            "z": float(body_rate_z[0])
        },
        "time": {
            "year": current_time.year,
            "month": current_time.month,
            "day": current_time.day,
            "hour": current_time.hour,
            "minute": current_time.minute,
            "second": current_time.second
        },
        "referenceVectors": {
            "sun": {
                "x": float(sun_vector_x[0]),
                "y": float(sun_vector_y[0]),
                "z": float(sun_vector_z[0])
            },
            "nadir": {
                "x": float(nadir_vector_x[0]),
                "y": float(nadir_vector_y[0]),
                "z": float(nadir_vector_z[0])
            },
            "velocity": {
                "x": float(velocity_vector_x[0]),
                "y": float(velocity_vector_y[0]),
                "z": float(velocity_vector_z[0])
            }
        },
        "simulationSequence": simulation_sequence
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(telemetry_data, f, indent=2)
    
    print(f"Telemetry data generated for {duration_sec} seconds at {sampling_rate_hz} Hz")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate spacecraft telemetry data')
    parser.add_argument('-d', '--duration', type=int, default=300,
                        help='Duration of the simulation in seconds (default: 300s = 5 minutes)')
    parser.add_argument('-r', '--rate', type=int, default=1,
                        help='Sampling rate in Hz (default: 1 Hz)')
    parser.add_argument('-o', '--output', type=str, default="spacecraft_telemetry.json",
                        help='Output JSON file path (default: spacecraft_telemetry.json)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate telemetry
    generate_telemetry(
        duration_sec=args.duration,
        sampling_rate_hz=args.rate,
        output_file=args.output
    )