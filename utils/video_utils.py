import cv2 
from tqdm import tqdm 

def read_video(video_path):
    """ 
    Read a video, frame by frame.
    
    
    Args :
        video_path (string) : path to the video
    Returns :
        List : Frames of Video 
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        
        # If there is no next frame 
        if not ret:
            break 

        frames.append(frame)
    
    # Releasing the capture object (no frames left)
    cap.release()

    return frames



def save_video(output_video_frames, output_video_path):

    # Get shape of frames (since same video, just look at the first frame)
    height, width = output_video_frames[0].shape[0], output_video_frames[0].shape[1]

    # Set compression method 
    compression_method = cv2.VideoWriter_fourcc(*'mp4v') # instead of MJPG compression

    # Compress each frame, use fps = 60
    out = cv2.VideoWriter(output_video_path, compression_method, 60, (width, height))

    # Process each frame 
    for frame in tqdm(output_video_frames):
        out.write(frame)

    out.release()


