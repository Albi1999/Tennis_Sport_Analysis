# pip install moviepy==1.0.3

import cv2 
from tqdm import tqdm 
from moviepy.editor import VideoFileClip

def read_video(video_path): # TODO : read in fps I think so we can reuse 
    """ 
    Read a video, frame by frame.
    
    
    Args :
        video_path (string) : path to the video
    Returns :
        List : Frames of Video 
    """

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []

    while True:
        ret, frame = cap.read()
        
        # If there is no next frame 
        if not ret:
            break 

        frames.append(frame)
    
    # Releasing the capture object (no frames left)
    cap.release()

    return frames, fps 



def save_video(output_video_frames, output_video_path, fps):

    # Get shape of frames (since same video, just look at the first frame)
    height, width = output_video_frames[0].shape[0], output_video_frames[0].shape[1]

    # Set compression method 
    compression_method = cv2.VideoWriter_fourcc(*'mp4v') # instead of MJPG compression

    # Compress each frame
    out = cv2.VideoWriter(output_video_path, compression_method, fps, (width, height))

    # Process each frame 
    for frame in tqdm(output_video_frames):
        out.write(frame)

    out.release()





def convert_mp4_to_mp3(mp4_file, mp3_file):
    """
    Convert an MP4 file to MP3 format.
    
    Parameters:
    mp4_file (str): Path to the input MP4 file
    mp3_file (str): Path to the output MP3 file
    """

    # Load the video file
    video = VideoFileClip(mp4_file)
    
    # Extract the audio
    audio = video.audio
    
    # Write the audio to an MP3 file
    audio.write_audiofile(mp3_file)
    
    # Close the files to free up resources
    audio.close()
    video.close()


        



