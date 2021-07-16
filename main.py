import cv2
import mediapipe as mp
import numpy as np
import csv
from csv import DictWriter
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
# Curl counter variables
counter = 0
stage = None
fo = cv2.FONT_HERSHEY_SIMPLEX
co1 = (0, 255, 0)
co2 = (0, 0, 255)
fields = []
rows = []
x = 1 #random variable
er = 10 #error percent
r = 10 #radius of circle
rid = 1 # aasan number
with open('event.csv', 'r') as red:
    csvr = csv.reader(red)

    fields = next(csvr)

    for row in csvr:
        rows.append(row)
        if(x==rid):
         rarray = row
         print(rarray)
        x += 1 / 2

ls1 = float(rarray[1])
rs1 = float(rarray[2])
lh1 = float(rarray[3])
rh1 = float(rarray[4])
lk1 = float(rarray[5])
rk1 = float(rarray[6])
print(ls1,rs1,lh1,rh1,lk1,rk1)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[
        0])  # AM; Means theta = angle of slope 1 - angle of slope 2
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            knee_R = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            knee_L = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            ankle_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            ankle_L = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            # Calculate angle
            angle_bshoul_L = calculate_angle(shoulder_R, shoulder_L, elbow_L)
            angle_bshoul_R = calculate_angle(shoulder_L, shoulder_R, elbow_R)
            angle_ashoul_R = calculate_angle(elbow_R, shoulder_R, hip_R)
            angle_ashoul_L = calculate_angle(elbow_L, shoulder_L, hip_L)
            angle_ahip_L = calculate_angle(shoulder_L, hip_L, knee_L)
            angle_ahip_R = calculate_angle(shoulder_R, hip_R, knee_R)
            angle_aknee_L = calculate_angle(hip_L, knee_L, ankle_L)
            angle_aknee_R = calculate_angle(hip_R, knee_R, ankle_R)

            # Visualize angle
            cv2.putText(image, str(angle_ashoul_L),
                        tuple(np.multiply(shoulder_L, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(angle_ashoul_R),
                        tuple(np.multiply(shoulder_R, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(angle_ahip_R),
                        tuple(np.multiply(hip_R, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(angle_ahip_L),
                        tuple(np.multiply(hip_L, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(angle_aknee_R),
                        tuple(np.multiply(knee_R, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(angle_aknee_L),
                        tuple(np.multiply(knee_L, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            if (ls1 + er > angle_ashoul_L and ls1 - er < angle_ashoul_L):
                cv2.circle(image,tuple(np.multiply(shoulder_L, [640, 480]).astype(int)),r,co1,r)
            else:
                cv2.circle(image, tuple(np.multiply(shoulder_L, [640, 480]).astype(int)), r, co2, r)
            if (rs1 + er > angle_ashoul_R and rs1 - er < angle_ashoul_R):
                cv2.circle(image, tuple(np.multiply(shoulder_R, [640, 480]).astype(int)), r, co1, r)
            else:
                cv2.circle(image, tuple(np.multiply(shoulder_R, [640, 480]).astype(int)), r, co2, r)
            if (lh1 + er > angle_ahip_L and lh1 - er < angle_ahip_L):
                cv2.circle(image, tuple(np.multiply(hip_L, [640, 480]).astype(int)), r, co1, r)
            else:
                cv2.circle(image, tuple(np.multiply(hip_L, [640, 480]).astype(int)), r, co2, r)
            if (rh1 + er > angle_ahip_R and rh1 - er < angle_ahip_R):
                cv2.circle(image,tuple(np.multiply(hip_R, [640, 480]).astype(int)),r,co1,r)
            else:
                cv2.circle(image, tuple(np.multiply(hip_R, [640, 480]).astype(int)), r, co2, r)
            if (lk1 + er > angle_aknee_L and lk1 - er < angle_aknee_L):
                cv2.circle(image, tuple(np.multiply(knee_L, [640, 480]).astype(int)), r, co1, r)
            else:
                cv2.circle(image, tuple(np.multiply(knee_L, [640, 480]).astype(int)), r, co2, r)
            if (rk1 + er > angle_aknee_R and rk1 - er < angle_aknee_R):
                cv2.circle(image, tuple(np.multiply(knee_R, [640, 480]).astype(int)), r, co1, r)
            else:
                cv2.circle(image, tuple(np.multiply(knee_R, [640, 480]).astype(int)), r, co2, r)

        except:
            pass
        cv2.rectangle(image, (0, 0), (450, 35), (0, 0, 0, 0), -1)

        # Rep data
        cv2.putText(image, 'FitCore Engine Testing Edition (C) FitFrame 2021', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 29), 1, cv2.LINE_AA)

        # cv2.putText(image, str(counter),
        #            (10, 60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        # cv2.putText(image, 'STAGE', (65, 12),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, stage,
        #            (60, 60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        cv2.imshow('Mediapipe Feed', image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
          break



    cap.release()
    cv2.destroyAllWindows()
