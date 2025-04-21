import cv2
import mediapipe as mp
import math

def calculate_angle(a, b):
    # คำนวณองศาระหว่างจุด a กับ b
    angle = math.degrees(math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

# เริ่ม Mediapipe + OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

cap = cv2.VideoCapture(1)

with mp_pose.Pose(
                static_image_mode= False,    # หากกำหนดเป็น True จะประมวลผลทุกเฟรมเป็น detection ใหม่ เหมาะกับรูปนิ่ง
                model_complexity= 1,    # เลือกระดับความซับซ้อนของโมเดล (0 ไวสุด แต่ละเอียดน้อยสุด)
                enable_segmentation= False,  # ปิดใช้ human segmentation mask
                smooth_landmarks=True,
                min_detection_confidence=0.7,# เปิด smoothing เพื่อให้ landmark วิ่งนุ่มขึ้น
                min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            lm       = results.pose_landmarks.landmark
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip      = lm[mp_pose.PoseLandmark.LEFT_HIP]

            # **ตรงนี้** คำนวณมุมก่อนใช้
            angle = calculate_angle(shoulder, hip)

            # ตรวจ posture ช่วง 88°–92° (ปรับเลขได้ตามต้องการ)
            if 88.2 < angle < 99.3:
                posture = "Correct Posture"
                color   = (0,255,0)
            else:
                posture = "Incorrect Posture"
                color   = (0,0,255)

            # แสดงผล
            cv2.putText(frame, f"Angle: {int(angle)} deg",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, posture,
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow('Skeleton + Posture', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()