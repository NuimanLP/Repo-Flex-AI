# MediaPipe Pose Detection for Images

A powerful Python script for batch processing images with MediaPipe's pose detection capabilities. The script can detect human poses in images, visualize anatomical landmarks, determine posture correctness, and process multiple images in a single command.

## Features

- Process single images or batches of images in one command
- GPU acceleration for faster processing (Metal on macOS, CUDA on other platforms)
- Customizable resolution settings for different use cases
- Detailed pose visualization with colored landmarks and connections
- Posture analysis with angle measurement
- Command-line interface with multiple options
- Progress tracking for batch processing

## Requirements

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy (`pip install numpy`)
- A compatible GPU for acceleration (optional but recommended)

## Basic Usage

### Processing a Single Image

```bash
python mediapipe_img.py --image path/to/image.jpg --output path/to/output.jpg
```

### Processing Multiple Images

```bash
python mediapipe_img.py --images image1.jpg image2.jpg image3.jpg --output output_directory/
```

### Processing an Entire Directory of Images

```bash
python mediapipe_img.py --dir input_directory/ --output output_directory/
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--image` | Path to a single input image |
| `--images` | Paths to multiple input images (space-separated) |
| `--dir` | Directory containing input images to process |
| `--output` | Path for output: file path for single image or directory for multiple images |
| `--output-dir` | Alternative directory for saving multiple output images |
| `--show` | Display processed images (may be slow for many images) |
| `--resolution` | Resolution preset to use (low, medium, hd, full_hd, custom) |

## Resolution Options

The script supports different resolution presets that affect both processing speed and detection quality:

| Preset | Resolution | Description |
|--------|------------|-------------|
| `low` | 640x480 | Fast processing, less detail |
| `medium` | 800x600 | Balanced performance and quality |
| `hd` | 1280x720 | Good detail with reasonable performance (default) |
| `full_hd` | 1920x1080 | High detail but slower processing |
| `custom` | 1500x1200 | Very high detail, slowest processing |

Example:
```bash
python mediapipe_img.py --dir images/ --output results/ --resolution hd
```

## Model Complexity

The script currently uses the "lite" model variant (`pose_landmarker_lite.task`), which offers a good balance between speed and accuracy. MediaPipe also offers:

- **Lite model**: Fast processing, good for real-time applications
- **Full model**: Better accuracy, larger model size, slower processing
- **Heavy model**: Highest accuracy, largest model size, slowest processing

To use a different model complexity:

1. Download the desired model:
   - Full: [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
   - Heavy: [pose_landmarker_heavy.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task)

2. Place the downloaded model in the same directory as the script

3. Update the `MODEL_PATH` variable in the script to point to your desired model

## Example Use Cases

### Fast Processing for Many Images

```bash
python mediapipe_img.py --dir many_images/ --output results/ --resolution low
```

### Highest Quality Detection

```bash
python mediapipe_img.py --dir important_images/ --output detailed_results/ --resolution full_hd
```

### Interactive Mode with Image Display

```bash
python mediapipe_img.py --dir images/ --output results/ --show
```

### Process Images with Specific Naming Pattern

```bash
python mediapipe_img.py --images photo1.jpg photo2.jpg photo5.jpg --output specific_results/
```

## Output Format

Processed images will include:
- Skeletal visualization with colored joints and connections
- Posture angle calculation
- Detection status information
- GPU acceleration status

Each processed image is saved with `_processed` appended to the original filename:
- `original.jpg` → `original_processed.jpg`

## Troubleshooting

- If you encounter "GPU acceleration failed" messages, the script will automatically fall back to CPU processing
- For "Model file not found" errors, the script uses the bundled MediaPipe model
- If no pose is detected in an image, the image will be processed but no skeleton will be drawn

## Performance Considerations

- Higher resolutions provide better detection quality but slower processing
- Using a compatible GPU significantly improves processing speed
- When processing many images, avoid using the `--show` option to maximize throughput
- For best performance on macOS, Metal acceleration is used automatically

# Posture Detection Program

## What This Program Does
- Uses your webcam to detect and monitor your posture in real-time
- Provides visual feedback of your posture on screen
- Plays audio alerts when incorrect posture is detected
- Shows angle measurements to help you maintain correct posture
- Includes fall detection with emergency alerts

## Prerequisites
- A computer with a working webcam
- Python 3.6 or higher
- Git (for downloading the program)
- Internet connection (for installing packages)

## Complete Setup Guide for Beginners

### 1. Install Required Software

#### Install Python
1. Visit [Python Downloads](https://www.python.org/downloads/)
2. Download the latest version for your system (Windows/Mac)
3. **IMPORTANT:** During installation, CHECK the box that says "Add Python to PATH"
4. To verify installation:
   - Open Terminal (Mac) or Command Prompt (Windows)
   - Type: `python --version`
   - You should see something like "Python 3.11.0"

#### Install Git
1. Visit [Git Downloads](https://git-scm.com/downloads)
2. Download and install for your system
3. To verify installation:
   - Open Terminal/Command Prompt
   - Type: `git --version`
   - You should see something like "git version 2.30.1"

### 2. Download and Set Up the Program

#### Open Terminal or Command Prompt
- **Mac**:
  - Press Command (⌘) + Space
  - Type "Terminal"
  - Press Enter
- **Windows**:
  - Press Windows key
  - Type "cmd"
  - Press Enter

#### Create a Project Folder
```
cd Documents
mkdir PoseDetection
cd PoseDetection
```

#### Get the Program Files
```
git clone [repository-URL]
cd MediapipeTestSkel
```

#### Set Up Python Virtual Environment
Create the virtual environment:
```
python -m venv venv
```

Activate the virtual environment:
- **Mac/Linux**:
  ```
  source venv/bin/activate
  ```
- **Windows**:
  ```
  venv\Scripts\activate
  ```

You should see `(venv)` at the beginning of your command line.

#### Install Required Packages
- Install the main packages:
  ```
  pip install opencv-python>=4.5.0 mediapipe>=0.8.10 numpy>=1.20.0 pygame
  ```

- **For Mac Users Only** (additional required package):
  ```
  pip install PyObjC
  ```

### 3. Running the Program
```
python mediapipe_pose.py
```

### Camera Selection
The program is currently set to use camera index 1 (secondary camera). If you only have one camera or want to use your primary camera:
1. Open `mediapipe_pose.py` in a text editor
2. Find the line: `cap = cv2.VideoCapture(1)`
3. Change it to: `cap = cv2.VideoCapture(0)`

### Program Files Structure
```
MediapipeTestSkel/
├── audio/                     # Contains alert sound files
│   ├── EX_accident.mp3       
│   ├── EX_incorrect.mp3      # Main alert sound
│   └── error-170796.mp3
├── mediapipe_pose.py         # Main program file
├── requirements.txt          # Package dependencies
└── venv/                     # Virtual environment
```

### Audio Alerts
The program uses audio files in the `audio/` folder for posture alerts:
- `EX_incorrect.mp3`: Plays when incorrect posture is detected
- `EX_accident.mp3`: Plays when a fall is detected
- Make sure your system's audio is enabled to hear the alerts

### Features
1. Posture Detection
   - Monitors the angle between your shoulder and hip
   - Correct posture is maintained when the angle is between 78° and 112.3°
   - Visual feedback shows your current angle and posture status
   - Audio alert plays when posture is incorrect

2. Fall Detection
   - Automatically detects if a person has fallen
   - Triggers an alert sound when a fall is detected
   - Uses nose position and shoulder orientation for detection

## Usage Instructions
- When the program starts, it will open a window showing your webcam feed
- Stand in front of the camera so your upper body is visible
- The program will:
  - Draw a skeleton overlay on your body
  - Show your current posture angle
  - Display "Correct Posture" (green) or "Incorrect Posture" (red)
  - Play an alert sound when your posture is incorrect
  - Display "Fall Detected!" and play a different alert when a fall is detected
- To exit the program, press the ESC key

## Troubleshooting

### Python Command Not Found
- Ensure you checked "Add Python to PATH" during installation
- Try closing and reopening Terminal/Command Prompt
- On Windows, try using `py` instead of `python`

### Package Installation Issues
- Make sure you're connected to the internet
- Ensure the virtual environment is activated (you see `(venv)` in terminal)
- Try updating pip first:
  ```
  python -m pip install --upgrade pip
  ```

### Camera Not Working
- Check if your webcam is properly connected
- Allow camera permissions when prompted
- On Mac: System Preferences -> Security & Privacy -> Camera
- Try a different camera if available

### Audio-Related Errors
- Make sure your system's audio is working
- Check that the audio files exist in the `audio/` folder
- Try restarting the program

## Important Reminders
- Keep the Terminal/Command Prompt window open while running the program
- The virtual environment must be activated (showing `(venv)`) when running the program
- You need a working webcam for the program to function

