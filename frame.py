import cv2

def extract_frame(video_path, output_image_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the specified frame
    ret, frame = cap.read()
    
    if ret:
        # Save the frame as a JPG file
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
    else:
        print(f"Error: Could not read frame {frame_number}.")
    
    # Release the video capture object
    cap.release()

# Define input and output paths
video_path = 'output/final/output_video101.mp4'
output_image_path = 'data/output_frame2.jpg'
frame_number = 103  # Change this to select a different frame

# Extract and save the specified frame
extract_frame(video_path, output_image_path, frame_number)

