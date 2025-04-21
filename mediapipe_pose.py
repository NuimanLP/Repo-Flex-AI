import cv2
import mediapipe as mp
import math
import time
import pygame  # ใช้ pygame เล่นไฟล์เสียง

# --- เตรียม pygame mixer ---
pygame.mixer.init()

def calculate_angle(a, b):
    """คำนวณองศาระหว่างจุด a กับ b"""
    angle = math.degrees(math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

# --- ตั้งค่าความละเอียดหน้าต่างและกล้อง ---
FRAME_WIDTH  = 800
FRAME_HEIGHT = 650

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

window_name = 'Skeleton + Posture'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

alert_sound = pygame.mixer.Sound('audio/EX_incorrect.mp3')
alert_sound.set_volume(0.5)   # 0.0 = ปิดเสียง, 1.0 = ดังสุด

# ตัวแปรควบคุมการเล่นเสียงไม่ให้ถี่เกินไป
last_alert_time = 0
alert_interval = 3   # วินาที

# เตรียม MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # พลิกภาพเหมือนกระจก และแปลงสีไปเป็น RGB
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ประมวลผลหา pose
        results = pose.process(rgb)

        if results.pose_landmarks:
            # วาด skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # ดึง landmark
            lm       = results.pose_landmarks.landmark
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip      = lm[mp_pose.PoseLandmark.LEFT_HIP]
            angle    = calculate_angle(shoulder, hip)

            # ตรวจ posture ช่วง 90°–112.3°
            if 78 < angle < 112.3:
                posture = "Correct Posture"
                color   = (0,255,0)
            else:
                posture = "Incorrect Posture"
                color   = (0,0,255)

                # เล่นเสียงเตือนเมื่อผิดท่า และครบช่วงดีเลย์
                now = time.time()
                if now - last_alert_time > alert_interval:
                    alert_sound.play()
                    last_alert_time = now

            # แสดงมุมและสถานะบนภาพ
            cv2.putText(frame, f"Angle: {angle:.1f}°",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, posture,
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # แสดงผล
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
            break

# ปิดทุกอย่างเมื่อจบ
cap.release()
cv2.destroyAllWindows()