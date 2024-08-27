import cv2
import os

# Path to the folder containing the videos
videos_folder = 'Videos'

# Path to the output folder where extracted frames will be saved
output_dataset_folder = 'dataset'
os.makedirs(output_dataset_folder, exist_ok=True)

# Iterate over all video files in the Videos folder
for video_file in os.listdir(videos_folder):
    # Get the full path to the video file
    video_path = os.path.join(videos_folder, video_file)

    # Check if the file is a video (e.g., .mp4)
    if os.path.isfile(video_path) and video_file.endswith('.mp4'):
        # Get the base name of the video file (without the extension)
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        # Create a folder to store the extracted frames with the same name as the video inside the dataset folder
        output_folder = os.path.join(output_dataset_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the interval at which to extract frames
        interval = total_frames // 20

        # Extract and save 20 frames at different intervals
        for i in range(20):
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)

            # Read the frame
            ret, frame = cap.read()

            if ret:
                # Save the frame as an image file
                frame_name = f'frame_{i + 1}.jpg'
                cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            else:
                print(f"Failed to read frame at index {i * interval} in video '{video_name}'")

        # Release the video capture object
        cap.release()

print(f"Frames have been successfully extracted and saved in the '{output_dataset_folder}' folder.")
