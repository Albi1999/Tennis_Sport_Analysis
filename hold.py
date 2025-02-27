
AUDIO = True

device = "cuda" if torch.cuda.is_available() else "cpu"

video_number = 101
input_video_path = f'data/input_video{video_number}.mp4'  # Toy example
if AUDIO:
    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
#input_video_path = f'data/videos/video_{video_number}.mp4' # Real example
output_video_path = f'output/output_video{video_number}.mp4'



# Detect Ball 
if ball_tracker_method == 'yolo':
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                read_from_stub = True,
                                                stub_path = 'tracker_stubs/ball_detections.pkl')
    # Interpolate the missing tracking positions for the ball
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    if AUDIO:
        ball_sound_detections = ball_tracker.get_ball_shot_frames_audio(input_video_path_audio)




from ultralytics import YOLO
import cv2
import pickle 
import pandas as pd 
import librosa 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfilt




    def get_ball_shot_frames_audio(self, audio_file, plot = True):
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Apply bandpass filter (150Hz-1800Hz)
        nyquist = 0.5 * sr
        low = 150 / nyquist
        high = 1800 / nyquist
        sos = butter(N=6, Wn=[low, high], btype='band', output='sos')
        y_filtered = sosfilt(sos, y)
        
        # Compute the envelope of the filtered signal
        y_abs = np.abs(y_filtered)
        
        # Apply smoothing to the envelope (adjust window_size as needed)
        window_size = int(0.01 * sr)  # 10ms window
        y_envelope = np.convolve(y_abs, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks in the envelope
        # Lower height threshold to catch more peaks
        peaks, _ = find_peaks(y_envelope, 
                            height=0.02,  # Lower threshold to catch more peaks
                            distance=int(0.3 * sr),  # Minimum distance between peaks
                            prominence=0.01)  # Find all distinct peaks 
        
        # Convert peak positions to time (seconds)
        hit_times = peaks / sr
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Plot filtered waveform with detected hits
            plt.subplot(2, 1, 1)
            times = np.linspace(0, len(y_filtered)/sr, len(y_filtered))
            plt.plot(times, y_filtered)
            plt.vlines(hit_times, -0.2, 0.2, color='r', linewidth=1)
            plt.title('Filtered Audio Waveform (150Hz-1800Hz) with Detected Hits')
            plt.xlabel('Time (s)')
            
            # Plot the envelope with detected peaks
            plt.subplot(2, 1, 2)
            plt.plot(times, y_envelope)
            plt.vlines(hit_times, 0, np.max(y_envelope), color='r', linewidth=1, label='Detected Hits')
            plt.title('Signal Envelope with Detected Peaks')
            plt.xlabel('Time (s)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("AUDIO.png")
           # plt.show()
        
        print(f"Detected {len(hit_times)} racket hits")
        return hit_times, y_filtered