import torch
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
import librosa 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfilt
from collections import defaultdict
from utils import euclidean_distance


device = "cuda" if torch.cuda.is_available() else "cpu"


def postprocess(feature_map, scale=2):
    # Scaling factor is dependent on original video width & height
    scaling_x, scaling_y = scale 
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
    x,y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scaling_x
            y = circles[0][0][1]*scaling_y
    return x, y


def infer_model(frames, model, scale):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """

    # Image size input for TrackNet is (360,640)
    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in tqdm(range(2, len(frames))):
        # Take 3 consecutive frames as input
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output, scale = scale)
        ball_track.append((x_pred, y_pred))

        # for not None values
        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)  
    return ball_track, dists 


def remove_outliers(ball_track, dists, max_dist = 100):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track



def write_track(frames, ball_track, trace = 7):

    output_video_frames = []
    for num in range(len(frames)):
        frame = frames[num].copy()
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=8-i)
                    
                else:
                    break
        output_video_frames.append(frame)
    return output_video_frames
    

import pandas as pd
import matplotlib.pyplot as plt
def get_ball_shot_frames_new(ball_positions_minicourt):
    """Based on change of direction in the mini court coordinates"""

    # TODO : see how we can optimize : the constants such as window & minimum_change_frames, is there smth better, maybe work with frames and 
    # how distant hits are from each other in a tennis match

    ball_positions = [x.get(1,[]) for x in ball_positions_minicourt]
    df_ball_positions = pd.DataFrame(ball_positions,columns=['x','y'])
    # Create a rolling window for the y positions
    df_ball_positions['y_rolling_mean'] = df_ball_positions['y'].rolling(window=3, min_periods=1, center=False).mean()
    df_ball_positions['delta_y'] = df_ball_positions['y_rolling_mean'].diff()
    df_ball_positions['ball_hit'] = 0

    plt.plot(df_ball_positions['delta_y'])
    plt.savefig("SHOTS.png")

    minimum_change_frames_for_hit = 20 # for this amount of frames it has to keep the changed direction
                                       # to be considered a hit
    for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
        negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
        positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

        if negative_position_change or positive_position_change:
            change_count = 0 
            for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                if negative_position_change and negative_position_change_following_frame:
                    change_count+=1
                elif positive_position_change and positive_position_change_following_frame:
                    change_count+=1
        
            if change_count>minimum_change_frames_for_hit-1:
                df_ball_positions['ball_hit'].iloc[i] = 1

    hit_frames = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

    return hit_frames
    



def get_ball_shot_frames_audio(audio_file, fps, plot=True):
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
    
    # Convert times to frame numbers
    hit_frames = [int(round(time * fps)) for time in hit_times]
    
    if plot:
        plt.figure(figsize=(12, 10))
        
        # Plot filtered waveform with detected hits
        plt.subplot(3, 1, 1)
        times = np.linspace(0, len(y_filtered)/sr, len(y_filtered))
        plt.plot(times, y_filtered)
        plt.vlines(hit_times, -0.2, 0.2, color='r', linewidth=1)
        plt.title('Filtered Audio Waveform (150Hz-1800Hz) with Detected Hits')
        plt.xlabel('Time (s)')
        
        # Plot the envelope with detected peaks
        plt.subplot(3, 1, 2)
        plt.plot(times, y_envelope)
        plt.vlines(hit_times, 0, np.max(y_envelope), color='r', linewidth=1, label='Detected Hits')
        plt.title('Signal Envelope with Detected Peaks')
        plt.xlabel('Time (s)')
        plt.legend()
        
        # Plot frame numbers
        plt.subplot(3, 1, 3)
        frame_times = np.arange(0, len(y_filtered)/sr, 1/fps)
        frame_indices = np.arange(0, len(frame_times))
        if len(frame_times) > 1000:  # If too many frames, subsample for clarity
            step = len(frame_times) // 1000
            frame_times = frame_times[::step]
            frame_indices = frame_indices[::step]
        plt.plot(frame_times, frame_indices, 'b-', alpha=0.5)
        plt.scatter([hit_times], [hit_frames], color='r', s=50)
        for i, (t, f) in enumerate(zip(hit_times, hit_frames)):
            plt.annotate(f"Frame {f}", (t, f), xytext=(5, 5), textcoords='offset points')
        plt.title('Detected Hits by Frame Number')
        plt.xlabel('Time (s)')
        plt.ylabel('Frame Number')
        
        plt.tight_layout()
        plt.savefig("AUDIO.png")
        # plt.show()
    

    return hit_frames 


def combine_visual_audio(ball_shots_frames, ball_shots_frames_audio, fps):
    """ 
    Idea is :
        The first hit in each game is very loud and not registered by the "change of direction"
        logic --> therefore we detect it with audio. Even if the game starts in the middle,
        since we work on a set, it should still be okay, but with this we catch the possible
        first hit.
        Audio signals give us a more clear point of where the ball was really hit : Therefore
        we want to use these frames as our reference points
    """
    
    # use a set to not have any duplicates
    ball_shots_frames_final = set()

    # Add initial hit
    ball_shots_frames_final.add(ball_shots_frames_audio[0])

    # We now iterate through the found direction changes and see if we can find
    # a close frame hit also in the audio
    for i in ball_shots_frames:
        for j in ball_shots_frames_audio:
            if i - 10 < j < i + 10:
                ball_shots_frames_final.add(j)

    # Next, we need to take into account that there will be more silent hits that our audio model might not recognize, but are recognized by the change of direction model
    # Therefore, we check if we have large gaps in the audio results where we have some results of the change of direction model

    # In a "normal" ball exchange, we can say that there is roughly one hit per second (# TODO : check), thus in frames exactly our fps 
    # Therefore, we check for each value in ball_shots_frames, if there is no frame in ball_shots_frames_audio in fps distance, then we should also take that frame
    # TODO : implement



    
    return ball_shots_frames_final





def draw_ball_hits(video_frames, hit_frames):

    output_video_frames = []
    counter = 0
    for i,frame in enumerate(video_frames):
        cv2.putText(frame, f"Racket Hit n. {counter}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        if i in hit_frames:
            counter += 1

    
        output_video_frames.append(frame)
    
    return output_video_frames 

def convert_ball_detection_to_bbox(ball_track, padding=5):
    """ Convert ball detection to bounding box format, similar as in YOLO implementation.
        Therefore, we want to return a list of dictionaries (one for each frame), with
        format 1 : [x_min, y_min, x_max, y_max]"""
    
    lst_of_bboxes = []
    
    # Iterate over all TrackNet (x,y) coordinates
    # len(ball_track) --> number of frames

    for i in range(len(ball_track)):
        bboxes = {}
        if ball_track[i][0]:
            x = ball_track[i][0]
            y = ball_track[i][1]
            # key 1 and values x_min, y_min, x_max, y_max
            bboxes[1] = [x-padding, y-padding, x+padding, y+padding]
        else: # Draw somewhere out of image # TODO : check if works / make more consistent
            bboxes[1] = [6000,6000,6000,6000]
        
        lst_of_bboxes.append(bboxes)

    return lst_of_bboxes