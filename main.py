import cv2
import mediapipe as mp
import numpy as np
import helper

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils  # for drawing the key points (joints)
    mp_pose = mp.solutions.pose  # extract the pose (key points)

    cap = cv2.VideoCapture(1)
    ## Set up mediapipe instance
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
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = helper.calculate_angle(left_shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Calculate Distance between two shoulder points to ensure shoulder is pointing to camera
                shoulder_dist = helper.calc_dist(left_shoulder, right_shoulder)

                height, width = image.shape[:2]

                # Define the position for the text (bottom right corner)
                position = (width - 600, height - 10)  # Adjust these values as needed

                # Check the value of shoulder_dist and set the text and color
                if shoulder_dist > 0.2:
                    text = f'FACE YOUR SHOULDER TO THE CAMERA. Dist = {shoulder_dist}'
                    color = (0, 0, 255)  # Red color in BGR
                else:
                    text = f'GOOD CAMERA PLACEMENT.'
                    color = (0, 255, 0)  # Green color in BGR

                # Use cv2.putText to put text on the frame
                font_scale = 0.6
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2

                cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


            except:
                pass

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