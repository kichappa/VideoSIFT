import numpy as np
import cv2

# path_to_video = 'assets/videos/DJI_20240329_154936_17_null_beauty.mp4'
path_to_video = 'assets/videos/20241203_000635.mp4'
cap = cv2.VideoCapture(path_to_video)

video_name = path_to_video.split('/')[-1]
print(f"Processing video \"{video_name}\"...")

images = []

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    images.append(frame)
    
    # If the frame is not read correctly, break the loop
    if not ret:
        break
    
    if len(images) > 0:
        # save the frame
        cv2.imwrite(f'assets/images/{video_name}_frame_{len(images)}.png', frame)
    
    
    if len(images) == 0+128:
        break