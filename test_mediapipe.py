#!/usr/bin/env python3
import mediapipe as mp
import os
import sys
import tensorflow as tf
from mediapipe.tasks.python.vision import RunningMode  # Updated import path

def test_mediapipe_support():
    print(f"MediaPipe version: {mp.__version__}")
    
    print("\nPython and TensorFlow Information:")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    
    print("\nAvailable Solutions:")
    for attr in dir(mp.solutions):
        if not attr.startswith('__'):
            print(f"- {attr}")
    
    print("\nTasks API Support:")
    try:
        from mediapipe.tasks.python import vision
        print("✓ Tasks API is available")
        
        # List available task modules
        print("\nTasks Modules:")
        for module in ["vision", "audio", "text", "core"]:
            try:
                __import__(f"mediapipe.tasks.python.{module}")
                print(f"- {module}: Available")
            except ImportError:
                print(f"- {module}: Not available")
                
    except ImportError as e:
        print(f"✗ Tasks API is not available: {e}")

    print("\nGPU Support:")
    # Check if TensorFlow can see the GPU
    print("TensorFlow devices:")
    physical_devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"- Physical devices: {physical_devices}")
    print(f"- GPU devices: {gpu_devices}")
    
    try:
        # Using correct import paths for Tasks API
        # Using correct import paths for Tasks API
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
        
        print("✓ MediaPipe Tasks API classes are available")
        try:
            # Create test options to verify
            base_options = BaseOptions(model_asset_path='pose_landmarker.task')
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE
            )
            print("✓ Successfully created PoseLandmarker options")
        except Exception as e:
            print(f"ℹ️ Cannot create full options (may need model file): {e}")
        
        # On macOS, Metal is used for acceleration
        if sys.platform == 'darwin':
            print("ℹ️ On macOS, GPU acceleration is handled through Metal")
            if gpu_devices:
                print("✓ Metal device is available for acceleration")
                print(f"✓ Found {len(gpu_devices)} Metal device(s)")
                for device in gpu_devices:
                    print(f"  - {device.name}")
            else:
                print("✗ No Metal device found for acceleration")
        
    except Exception as e:
        print(f"✗ Tasks API error: {str(e)}")

if __name__ == "__main__":
    test_mediapipe_support()

