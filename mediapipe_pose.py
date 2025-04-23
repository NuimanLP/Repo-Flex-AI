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

window_name = 'Skeleton + Posture + Fall Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

# --- โหลดไฟล์เสียง --- 
incorrect_sound = pygame.mixer.Sound('audio/EX_incorrect.mp3')
incorrect_sound.set_volume(0)
accident_sound  = pygame.mixer.Sound('audio/EX_accident.mp3')
accident_sound.set_volume(0.5)

# ตัวแปรควบคุมการเล่นเสียงไม่ให้ถี่เกินไป
last_incorrect_time = 0
incorrect_interval = 3   # วินาที
last_accident_time  = 0
accident_interval   = 5  # วินาที

# เตรียม MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # พลิกภาพ + แปลงสี
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            lm = results.pose_landmarks.landmark

            # --- Posture Check (ไหล่ซ้าย-สะโพกซ้าย) ---
            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip      = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            angle    = calculate_angle(shoulder, hip)

            if 78 < angle < 112.3:
                posture = "Correct Posture"
                color   = (0,255,0)
            else:
                posture = "Incorrect Posture"
                color   = (0,0,255)

                now = time.time()
                if now - last_incorrect_time > incorrect_interval:
                    incorrect_sound.play()
                    last_incorrect_time = now

            # --- Fall Detection (ล้ม) ---
            # ใช้ landmark NOSE + orientation ของบ่า
            nose = lm[mp_pose.PoseLandmark.NOSE]
            # ระดับ y ปกติอยู่สูงกว่า ~0.2–0.6; ถ้า nose.y>0.8 แสดงว่าศีรษะใกล้พื้น
            # และตรวจว่าลำตัวนอนราบ: มุมระหว่างบ่า L-R น้อยกว่า 30°
            sl = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            sr = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # มุมแนวนอนของบ่า
            horiz_angle = abs(math.degrees(math.atan2(sr.y - sl.y, sr.x - sl.x)))

            fall_detected = False
            if nose.y > 0.8 and horiz_angle < 30:
                fall_detected = True
                now2 = time.time()
                if now2 - last_accident_time > accident_interval:
                    accident_sound.play()
                    last_accident_time = now2

            # --- แสดงผลบนภาพ ---
            cv2.putText(frame, f"Angle: {angle:.1f}°",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, posture,
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if fall_detected:
                cv2.putText(frame, "Fall Detected!",
                            (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # แสดงผล
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC ออก
            break

# ปิดทั้งหมด
cap.release()
cv2.destroyAllWindows()