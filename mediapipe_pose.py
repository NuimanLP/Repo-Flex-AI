import cv2
import math
import time
import sys
import os
import pygame  # Use pygame for audio playback
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.core.base_options import BaseOptions  # Correct import path
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe.framework.formats import landmark_pb2

# Get the PoseLandmarker class and running mode
PoseLandmarker = pose_landmarker.PoseLandmarker
RunningMode = vision_task_running_mode.VisionTaskRunningMode

# Global variables for results
last_pose_results = None

# Callback function for LIVE_STREAM mode
def pose_result_callback(result, output_image, timestamp_ms):
    global last_pose_results
    last_pose_results = result

# Initialize pygame for audio
pygame.mixer.init()

# --- Load and set sound volumes ---
incorrect_sound = pygame.mixer.Sound('audio/EX_incorrect.mp3')
incorrect_sound.set_volume(0.5)  # Adjust volume (0.0 to 1.0)
accident_sound = pygame.mixer.Sound('audio/EX_accident.mp3')
accident_sound.set_volume(0.5)   # Adjust volume (0.0 to 1.0)

def calculate_angle(a, b):
    """Calculate angle between points a and b"""
    return abs(math.degrees(math.atan2(a.y - b.y, a.x - b.x)))

# Window configuration
window_name = "Posture Detection"
FRAME_WIDTH = 800
FRAME_HEIGHT = 650

# Camera setup
CAMERA_INDEX = 1  # Try 1 if 0 doesn't work
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Camera index {CAMERA_INDEX} failed, trying index 1...")
    CAMERA_INDEX = 1
    cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Create window
cv2.namedWindow(window_name)
cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

# Initialize MediaPipe Pose with GPU acceleration
print("Initializing MediaPipe Pose with GPU acceleration...")
is_macos = sys.platform == 'darwin'
gpu_backend = "Metal" if is_macos else "CUDA"
print(f"Using {gpu_backend} acceleration")

# Check if model file exists and download if needed
MODEL_PATH = 'pose_landmarker_lite.task'

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
    print("Using default model from MediaPipe...")
    MODEL_PATH = None  # MediaPipe will use bundled model

# Create PoseLandmarker with GPU delegate
try:
    # In MediaPipe 0.10.21, GPU acceleration is automatic on supported platforms
    base_options = BaseOptions(
        model_asset_path=MODEL_PATH
    )
    landmarker_options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.LIVE_STREAM,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        result_callback=pose_result_callback  # Add callback for LIVE_STREAM mode
    )
    pose_landmarker = PoseLandmarker.create_from_options(landmarker_options)
    print(f"MediaPipe Pose initialized with acceleration")
except Exception as e:
    print(f"Failed to initialize pose landmarker: {e}")
    print("Trying with bundled model...")
    # Try with bundled model
    base_options = BaseOptions(
        model_asset_path=None  # Use bundled model
    )
    landmarker_options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.LIVE_STREAM,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        result_callback=pose_result_callback  # Add callback for LIVE_STREAM mode
    )
    pose_landmarker = PoseLandmarker.create_from_options(landmarker_options)
    print("MediaPipe Pose initialized with bundled model")

# Initialize timing variables
last_incorrect = last_fall = 0
incorrect_interval, fall_interval = 3, 5  # seconds

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame from camera")
        break

    # Prepare frame
    # Prepare frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    
    # Create MediaPipe Image object correctly
    try:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
    except AttributeError:
        # Fallback for newer MediaPipe versions
        mp_image = mp_tasks.vision.Image(
            image_format=mp_tasks.vision.ImageFormat.SRGB,
            data=frame_rgb
        )
    # Run inference
    # Run inference with proper timestamp for LIVE_STREAM mode
    try:
        timestamp_ms = int(time.time() * 1000)  # Current timestamp in milliseconds
        # In LIVE_STREAM mode, detect_async doesn't return results directly
        # Results come through the callback function
        pose_landmarker.detect_async(mp_image, timestamp_ms)
        
        # Use the results from the callback
        results = last_pose_results
        if results is None:
            # Skip this frame if no results yet
            continue
            
    except Exception as e:
        print(f"Detection error: {e}")
        continue
    
    # Process landmarks if detected
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        # Draw landmarks
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * FRAME_WIDTH)
            y = int(landmark.y * FRAME_HEIGHT)
            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

        # Check posture
        lm = results.pose_landmarks[0]
        
        # Get landmark indices
        LEFT_SHOULDER = 11  # PoseLandmark.LEFT_SHOULDER
        LEFT_HIP = 23       # PoseLandmark.LEFT_HIP
        NOSE = 0            # PoseLandmark.NOSE
        RIGHT_SHOULDER = 12 # PoseLandmark.RIGHT_SHOULDER
        
        shoulder = lm[LEFT_SHOULDER]
        hip = lm[LEFT_HIP]
        angle = calculate_angle(shoulder, hip)
        if 78 < angle < 112:
            color, posture = (0,255,0), "Correct Posture"
        else:
            color, posture = (0,0,255), "Incorrect Posture"
            now = time.time()
            if now - last_incorrect > incorrect_interval:
                incorrect_sound.play()
                last_incorrect = now

        # Fall detection
        nose = lm[NOSE]
        left_shoulder = lm[LEFT_SHOULDER]
        right_shoulder = lm[RIGHT_SHOULDER]
        
        # Calculate horizontal angle of shoulders
        horiz_angle = abs(math.degrees(math.atan2(
            right_shoulder.y - left_shoulder.y, 
            right_shoulder.x - left_shoulder.x
        )))
        
        # Detect if fallen
        fall_detected = False
        if nose.y > 0.8 and horiz_angle < 30:  # Nose is low and shoulders nearly horizontal
            fall_detected = True
            now = time.time()
            if now - last_fall > fall_interval:
                accident_sound.play()
                last_fall = now
                
        # Display information
        cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, posture, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if fall_detected:
            cv2.putText(frame, "Fall Detected!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show GPU status
    cv2.putText(frame, f"{gpu_backend} Acceleration", 
               (FRAME_WIDTH - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display frame
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()
print("Done")
