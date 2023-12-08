import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Holistic module
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)

# Specify the folder containing images
image_folder = '/Users/prescott/dataset_examples'

# Specify the output CSV file path
csv_file_path = '/Users/prescott/dataset.csv'

# Iterate through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Assuming image files have these extensions
        image_path = os.path.join(image_folder, filename)
        print(f"Processing image: {image_path}")

        # Load an image from file
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = holistic.process(image_rgb)

        # Indication of Good or Bad Posture
        indication = ''
        if 'Good' in filename:
            indication = 'Good'
        else:
            indication = 'Bad'

        # Create a list to store data points for the selected landmark
        selected_landmarks = [indication]

        # Access and append specific landmarks to the list
        if results.pose_landmarks:
            # Indices of the selected landmarks
            # In order: FaceCenter, LeftShoulder, RightShoulder, LeftHip, RightHip
            selected_indices = [10, 11, 12, 23, 24]

            for idx in selected_indices:
                landmark = results.pose_landmarks.landmark[idx]
                selected_landmarks.extend([landmark.x, landmark.y, landmark.z])

        # Save selected landmarks to CSV file (append mode)
        with open(csv_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Check if the file is empty and write the header only if needed
            if csvfile.tell() == 0:
                csv_writer.writerow(['Good/Bad',
                                     'FaceCenter_X', 'FaceCenter_Y', 'FaceCenter_Z',
                                     'LeftShoulder_X', 'LeftShoulder_Y', 'LeftShoulder_Z',
                                     'RightShoulder_X', 'RightShoulder_Y', 'RightShoulder_Z',
                                     'LeftHip_X', 'LeftHip_Y', 'LeftHip_Z',
                                     'RightHip_X', 'RightHip_Y', 'RightHip_Z'])

            csv_writer.writerow(selected_landmarks)

print("Processing complete.")