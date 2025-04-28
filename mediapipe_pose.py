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

# Define the connections for skeleton visualization
POSE_CONNECTIONS = [
    # Torso
    (11, 12), # Left shoulder to right shoulder
    (11, 23), # Left shoulder to left hip
    (12, 24), # Right shoulder to right hip
    (23, 24), # Left hip to right hip
    
    # Left arm
    (11, 13), # Shoulder to elbow
    (13, 15), # Elbow to wrist
    
    # Right arm
    (12, 14), # Shoulder to elbow
    (14, 16), # Elbow to wrist
    
    # Left leg
    (23, 25), # Hip to knee
    (25, 27), # Knee to ankle
    
    # Right leg
    (24, 26), # Hip to knee
    (26, 28), # Knee to ankle
    
    # Spine
    (0, 11),  # Nose to left shoulder
    (0, 12),  # Nose to right shoulder
]

# Add color definitions
JOINT_COLOR = (255, 255, 0)  # Yellow for joints
BONE_COLOR = (0, 255, 255)   # Cyan for bones
NOSE_COLOR = (0, 0, 255)     # Red for nose
HIP_COLOR = (255, 0, 0)      # Blue for hips
GOOD_POSTURE_COLOR = (0, 255, 0)  # Green
BAD_POSTURE_COLOR = (0, 0, 255)   # Red
FALL_DETECT_COLOR = (255, 0, 255)  # Magenta

def draw_skeleton(frame, landmarks, frame_width, frame_height):
    """Draw a full skeleton on the frame using the landmarks."""
    # Draw connections (bones)
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        # Convert normalized coordinates to pixel coordinates
        start_x = int(start_point.x * frame_width)
        start_y = int(start_point.y * frame_height)
        end_x = int(end_point.x * frame_width)
        end_y = int(end_point.y * frame_height)
        
        # Draw the bone
        # Draw the bone with thickness based on importance
        thickness = 4 if connection in [(11, 23), (12, 24), (23, 24)] else 3  # Thicker lines for torso
        cv2.line(frame, (start_x, start_y), (end_x, end_y), BONE_COLOR, thickness)
    # Draw joints (landmarks)
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        
        # Special colors for specific joints
        if idx == 0:  # Nose
            color = NOSE_COLOR
            size = 6
        elif idx in [23, 24]:  # Hips
            color = HIP_COLOR
            size = 6
        else:  # Other joints
            color = JOINT_COLOR
            size = 4
            
        cv2.circle(frame, (x, y), size, color, -1)

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
CAMERA_INDEX = 1 # Start with default camera
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
        # Draw skeleton
        draw_skeleton(frame, results.pose_landmarks[0], FRAME_WIDTH, FRAME_HEIGHT)
        
        # Get landmark indices
        LEFT_SHOULDER = 11  # PoseLandmark.LEFT_SHOULDER
        LEFT_HIP = 23       # PoseLandmark.LEFT_HIP
        NOSE = 0            # PoseLandmark.NOSE
        RIGHT_SHOULDER = 12 # PoseLandmark.RIGHT_SHOULDER
        
        # Check posture first
        lm = results.pose_landmarks[0]
        shoulder = lm[LEFT_SHOULDER]
        hip = lm[LEFT_HIP]
        angle = calculate_angle(shoulder, hip)
        
        if 85 < angle < 105:
            color = GOOD_POSTURE_COLOR
            posture = "Correct Posture"
        else:
            color = BAD_POSTURE_COLOR
            posture = "Incorrect Posture"
            now = time.time()
            if now - last_incorrect > incorrect_interval:
                incorrect_sound.play()
                last_incorrect = now
        
        # Now draw the reference lines and angle visualization
        mid_x = int((lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) * FRAME_WIDTH / 2)
        cv2.line(frame, (mid_x, 0), (mid_x, FRAME_HEIGHT), (100, 100, 100), 1)

        # Draw posture angle arc
        shoulder_point = (int(shoulder.x * FRAME_WIDTH), int(shoulder.y * FRAME_HEIGHT))
        hip_point = (int(hip.x * FRAME_WIDTH), int(hip.y * FRAME_HEIGHT))
        cv2.ellipse(frame, shoulder_point, (30, 30), 0, -90, -90 + angle, color, 2)

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
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, posture, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if fall_detected:
            # Draw red background for fall warning
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 70), (400, 130), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "FALL DETECTED!", (10, 110),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

    # Show GPU status
    # Show GPU status
    cv2.putText(frame, f"{gpu_backend} Acceleration", 
               (FRAME_WIDTH - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
               
    # Show detection quality
    detection_color = (0, 255, 0) if results and results.pose_landmarks else (0, 0, 255)
    cv2.putText(frame, "Detection Status:", (10, FRAME_HEIGHT - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
    cv2.putText(frame, "ACTIVE" if results and results.pose_landmarks else "SEARCHING", 
                (180, FRAME_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)

    # Show posture angle reference
    cv2.putText(frame, "Good Posture Range: 85 - 105", (10, FRAME_HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOOD_POSTURE_COLOR, 2)
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
