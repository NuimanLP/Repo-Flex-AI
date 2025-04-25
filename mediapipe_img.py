import cv2
import math
import sys
import os
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.core.base_options import BaseOptions
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

# Window configuration
window_name = "Posture Detection"

# Resolution presets for different use cases
# These settings affect both processing and display:
# - Higher resolution = more detail but slower processing and more memory usage
# - Lower resolution = faster processing but less detail
RESOLUTION_PRESETS = {
    "low": (640, 480),      # Fast processing, less detail
    "medium": (800, 600),   # Balanced (original)
    "hd": (1280, 720),      # HD format, good detail
    "full_hd": (1920, 1080),# Full HD, high detail but slower
    "custom": (1500, 1200)  # Custom high resolution
}

# Default to HD resolution (good balance of detail and performance)
# NOTE: Higher resolutions may improve detection accuracy but will:
#  - Increase memory usage
#  - Slow down processing speed
#  - May not be necessary for all use cases
FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION_PRESETS["hd"]

# Uncomment ONE of these lines to change resolution:
# FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION_PRESETS["low"]      # Fast processing
# FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION_PRESETS["medium"]   # Original balanced setting
# FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION_PRESETS["full_hd"]  # High detail
# FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION_PRESETS["custom"]   # Custom high resolution

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

def calculate_angle(a, b):
    """Calculate angle between points a and b"""
    return abs(math.degrees(math.atan2(a.y - b.y, a.x - b.x)))

def process_image(image_path, output_path=None, show=False):
    """Process a single image and perform pose detection."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Resize the image to the desired dimensions
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    target_aspect = FRAME_WIDTH / FRAME_HEIGHT
    image_aspect = w / h
    
    if image_aspect > target_aspect:
        # Image is wider than target
        new_w = FRAME_WIDTH
        new_h = int(new_w / image_aspect)
    else:
        # Image is taller than target
        new_h = FRAME_HEIGHT
        new_w = int(new_h * image_aspect)
    
    # Resize using calculated dimensions
    image = cv2.resize(image, (new_w, new_h))
    
    # Create a blank canvas of target size
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Calculate position to center the image
    y_offset = (FRAME_HEIGHT - new_h) // 2
    x_offset = (FRAME_WIDTH - new_w) // 2
    
    # Place the resized image on the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image
    
    # Use the canvas as our working image
    image = canvas
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Pose with GPU acceleration
    print("Initializing MediaPipe Pose with GPU acceleration...")
    is_macos = sys.platform == 'darwin'
    gpu_backend = "Metal" if is_macos else "CUDA"
    print(f"Using {gpu_backend} acceleration")
    
    # Check if model file exists
    MODEL_PATH = 'pose_landmarker_full.task' #MODEL SELECT
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        print("Using default model from MediaPipe...")
        MODEL_PATH = None  # MediaPipe will use bundled model
    
    # Create the pose landmarker with SINGLE_IMAGE mode
    try:
        # Get the PoseLandmarker class and running mode
        PoseLandmarker = pose_landmarker.PoseLandmarker
        RunningMode = vision_task_running_mode.VisionTaskRunningMode
        
        base_options = BaseOptions(
            model_asset_path=MODEL_PATH
        )
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,  # Use IMAGE mode instead of LIVE_STREAM
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        detector = PoseLandmarker.create_from_options(options)
        print(f"MediaPipe Pose initialized with acceleration")
    except Exception as e:
        print(f"Failed to initialize pose landmarker: {e}")
        print("Trying with bundled model...")
        # Try with bundled model
        base_options = BaseOptions(
            model_asset_path=None  # Use bundled model
        )
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        detector = PoseLandmarker.create_from_options(options)
        print("MediaPipe Pose initialized with bundled model")
    
    # Create MediaPipe Image object
    try:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
    except AttributeError:
        # Fallback for newer MediaPipe versions
        mp_image = mp_tasks.vision.Image(
            image_format=mp_tasks.vision.ImageFormat.SRGB,
            data=image_rgb
        )
    
    # Run inference
    try:
        results = detector.detect(mp_image)
    except Exception as e:
        print(f"Detection error: {e}")
        detector.close()
        return
    
    # Process landmarks if detected
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        # Draw skeleton
        draw_skeleton(image, results.pose_landmarks[0], FRAME_WIDTH, FRAME_HEIGHT)
        
        # Get landmark indices
        LEFT_SHOULDER = 11  # PoseLandmark.LEFT_SHOULDER
        LEFT_HIP = 23       # PoseLandmark.LEFT_HIP
        RIGHT_SHOULDER = 12 # PoseLandmark.RIGHT_SHOULDER
        
        # Check posture
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
        
        # Draw the reference lines and angle visualization
        mid_x = int((lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) * FRAME_WIDTH / 2)
        cv2.line(image, (mid_x, 0), (mid_x, FRAME_HEIGHT), (100, 100, 100), 1)

        # Draw posture angle arc
        shoulder_point = (int(shoulder.x * FRAME_WIDTH), int(shoulder.y * FRAME_HEIGHT))
        hip_point = (int(hip.x * FRAME_WIDTH), int(hip.y * FRAME_HEIGHT))
        cv2.ellipse(image, shoulder_point, (30, 30), 0, -90, -90 + angle, color, 2)
                
        # Display information
        cv2.putText(image, f"Angle: {angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, posture, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Show GPU status
        cv2.putText(image, f"{gpu_backend} Acceleration", 
                   (FRAME_WIDTH - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                   
        # Show detection quality
        detection_color = (0, 255, 0) if results and results.pose_landmarks else (0, 0, 255)
        cv2.putText(image, "Detection Status:", (10, FRAME_HEIGHT - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
        cv2.putText(image, "DETECTED" if results and results.pose_landmarks else "NOT DETECTED", 
                    (180, FRAME_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)

        # Show posture angle reference
        cv2.putText(image, "Good Posture Range: 85 - 105", (10, FRAME_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOOD_POSTURE_COLOR, 2)
    else:
        print("No pose detected in the image.")
    
    # Display the result if show is True
    if show:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)  # Wait for a key press
    
    # Save the output if path is provided
    if output_path:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, image)
        return True  # Successfully saved
    
    # Clean up
    detector.close()
    if show:
        cv2.destroyAllWindows()
def process_multiple_images(image_paths, output_dir=None, show=False):
    """Process multiple images and save results to output directory."""
    total_images = len(image_paths)
    successful = 0
    
    print(f"Processing {total_images} images...")
    
    for i, image_path in enumerate(image_paths):
        # Skip if not an image file
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {image_path}")
            continue
            
        # Generate output path if output directory is provided
        output_path = None
        if output_dir:
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base}_processed{ext}")
        
        # Process the image and show progress
        print(f"[{i+1}/{total_images}] Processing: {image_path}")
        if process_image(image_path, output_path, show):
            successful += 1
    
    print(f"Processing complete. Successfully processed {successful} out of {total_images} images.")
    if output_dir:
        print(f"Results saved to: {output_dir}")

def get_image_files_from_directory(directory):
    """Get all image files from a directory."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = []
    
    for file in os.listdir(directory):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(directory, file))
    
    return image_files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='MediaPipe Pose Detection for Images')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to a single input image')
    input_group.add_argument('--images', type=str, nargs='+', help='Paths to multiple input images')
    input_group.add_argument('--dir', type=str, help='Directory containing input images')
    
    # Resolution option
    parser.add_argument('--resolution', type=str, choices=list(RESOLUTION_PRESETS.keys()), 
                        help=f'Resolution preset to use: {", ".join(RESOLUTION_PRESETS.keys())} (default: hd)')
    
    # Remove mutually exclusive group for output to allow --output to work with multiple images
    parser.add_argument('--output', type=str, help='Path to save output: file path for single image or directory for multiple images')
    parser.add_argument('--output-dir', type=str, help='Directory to save multiple output images (alternative to --output)')
    
    parser.add_argument('--show', action='store_true', help='Show processed images (may be slow for multiple images)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply resolution preset if specified
    global FRAME_WIDTH, FRAME_HEIGHT
    if args.resolution and args.resolution in RESOLUTION_PRESETS:
        FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION_PRESETS[args.resolution]
        print(f"Using {args.resolution} resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    
    # Determine the output directory for multiple images
    output_directory = None
    if args.output_dir:
        output_directory = args.output_dir
    elif args.output and (args.images or args.dir):
        # When --output is used with multiple images, treat it as a directory
        output_directory = args.output
    
    # Process single image
    if args.image:
        if args.output_dir:
            print("Error: For a single image use --output instead of --output-dir")
            return
        process_image(args.image, args.output, args.show)
    
    # Process multiple images from arguments
    elif args.images:
        if not output_directory and args.output:
            # Create output directory if needed
            output_directory = args.output
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                print(f"Created output directory: {output_directory}")
        
        process_multiple_images(args.images, output_directory, args.show)
    
    # Process all images in directory
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} is not a valid directory")
            return
            
        image_files = get_image_files_from_directory(args.dir)
        if not image_files:
            print(f"No image files found in {args.dir}")
            return
        
        if not output_directory and args.output:
            # Create output directory if needed
            output_directory = args.output
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                print(f"Created output directory: {output_directory}")
        
        process_multiple_images(image_files, output_directory, args.show)
if __name__ == "__main__":
    main()

