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
from copy import deepcopy
import pandas as pd
import pickle 



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


def remove_outliers(ball_track, dists, max_dist = 50):
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


def remove_outliers_final(ball_track, thresh = 150, consecutive_frames = 3):
    """
    
    After interpolation, still some outliers : we now
    check for consecutive frames where the ball
    is tracked closely to each other and then use
    this as the reference point for replacing
    outliers.
    
    """

    dists = []
    # Recalculate distances of each point pair
    for i in range(0, len(ball_track) - 1):
        # Check that it is not None
        if ball_track[i][0] and ball_track[i+1][0]:
            dist = euclidean_distance(ball_track[i], ball_track[i+1])
            dists.append(dist)
        else:
            dist = None 
            dists.append(None)

        if dist is not None and dist > thresh and i >= consecutive_frames:
            check_distances = [dists[x] for x in range(i - consecutive_frames, i)]
            # If there was any untracked distance (i.e. we lost tracking), we skip the 
            # iteration (because we could now have a large distance that is correct)
            if None in check_distances:
                continue
            check_distances = list(filter(lambda a : a is not None, check_distances))
            if ball_track[i] is not None and np.average(check_distances) <= thresh:
                ball_track[i + 1] = ball_track[i]
                # Reset the last calculated distance
                dists[-1] = 0




    return ball_track




def write_track(frames, ball_track, ball_shots_frames=[], trace = 7, draw_mode = 'circle'):

    # Use a set (because it has lookup in O(1))
    ball_shots_frames = set(ball_shots_frames)
    
    output_video_frames = []
    curr_rack_hit = -1000
    for num in range(len(frames)):
        frame = frames[num].copy()

                    


        # Draw trace of the ball
  
        # Store valid points we find
        valid_points = []
        # Save the closest current racket hit
        if num in ball_shots_frames:
            curr_rack_hit = num 
        # Collect valid points first
        for i in range(trace):
            if (num-i > 0) and ball_track[num-i][0]:
                # Check if there was a racket hit ; we want to reset the trace for each racket hit (such that is doesn't track the "weird"
                # patterns pre&post hit)
                # Only track starting every time from the racket hit 
                if (num-i) < curr_rack_hit:
                    pass 
                else:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    valid_points.append((x, y))
            else:
                break
        
        # Draw circles for all valid points
        if draw_mode == 'circle':
            for point in valid_points:
                frame = cv2.circle(frame, point, radius=2, color=(0, 255, 255), thickness=10)
        
        elif draw_mode == 'line':
            # Draw lines between consecutive points
            for i in range(1, len(valid_points)):
                frame = cv2.line(frame, valid_points[i], valid_points[i-1], color=(0, 0, 255), thickness=3)

        output_video_frames.append(frame)
    return output_video_frames


def detect_frames_TRACKNET(video_frames, video_number, tracker, video_width, video_height, read_from_stub, stub_path):


    if stub_path is not None and read_from_stub == True:
        with open(stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
        
    if stub_path is not None and read_from_stub == False: 
        # Calculate the correct scale factor for scaling back 
        # with TrackNet, we scaled to 640 width, 360 height
        scaling_x = video_width/640
        scaling_y = video_height/360
        ball_detections, dists = infer_model(video_frames, tracker, scale = (scaling_x, scaling_y))
        ball_detections = remove_outliers(ball_detections, dists)
        subtracks = split_track(ball_detections)
        for r in subtracks:
            ball_subtrack = ball_detections[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_detections[r[0]:r[1]] = ball_subtrack
    
        with open(stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)


    # Final removal of outliers based on distances after initial interpolation method
  #  ball_detections = remove_outliers_final(ball_detections, thresh= 300)
    

    # Copy TrackNet ball_detections
    ball_detections_tracknet = ball_detections.copy()
    ball_detections = convert_ball_detection_to_bbox(ball_detections)

    return ball_detections, ball_detections_tracknet

    

def get_ball_shot_frames_visual(ball_positions, fps, area):
    """Based on change of direction in the mini court coordinates"""


  #  ball_positions = [x.get(1,[]) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions,columns=['x','y'])
    df_ball_positions = df_ball_positions.iloc[area[0]:area[1]]
    # Create a rolling window for the y positions
    window_size = max(5, fps // 10)  # Adaptive window size based on fps
    df_ball_positions['y_rolling_mean'] = df_ball_positions['y'].rolling(window=window_size, min_periods=1, center=False).mean()
    df_ball_positions['delta_y'] = df_ball_positions['y_rolling_mean'].diff()
    df_ball_positions['ball_hit'] = 0

    plt.plot(df_ball_positions['delta_y'])
    plt.savefig("VISUAL.png")



    # Make this small to catch as much as possible
    minimum_change_frames_for_hit = max(5, fps//20)


    for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit)):
        negative_position_change = df_ball_positions['y_rolling_mean'].iloc[i] >0 and df_ball_positions['y_rolling_mean'].iloc[i+1] <0
        positive_position_change = df_ball_positions['y_rolling_mean'].iloc[i] <0 and df_ball_positions['y_rolling_mean'].iloc[i+1] >0

        if negative_position_change or positive_position_change:
            change_count = 0 
            for change_frame in range(i+1, i+int(minimum_change_frames_for_hit)+1):
                negative_position_change_following_frame = df_ball_positions['y_rolling_mean'].iloc[i] >0 and df_ball_positions['y_rolling_mean'].iloc[change_frame] <0
                positive_position_change_following_frame = df_ball_positions['y_rolling_mean'].iloc[i] <0 and df_ball_positions['y_rolling_mean'].iloc[change_frame] >0

                if negative_position_change and negative_position_change_following_frame:
                    change_count+=1
                elif positive_position_change and positive_position_change_following_frame:
                    change_count+=1
        
            if change_count>minimum_change_frames_for_hit-1:
                df_ball_positions['ball_hit'].iloc[i] = 1

    hit_frames = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

    return hit_frames




    

def get_ball_shot_frames_audio(audio_file, fps, plot=False):
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
                        distance=int(0.5 * sr),  # Minimum distance between peaks 
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




def get_ball_shot_frames_audio_refinement(audio_file, fps, frames_start=None, frames_end=None, 
                              peak_height=0.02, peak_prominence=0.01, peak_distance=0.5,
                              plot=False):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Convert frame range to time range if specified
    time_start = frames_start / fps if frames_start is not None else 0
    time_end = frames_end / fps if frames_end is not None else len(y) / sr
    
    # Convert time range to sample indices
    sample_start = int(time_start * sr)
    sample_end = min(int(time_end * sr), len(y))
    
    # Extract the relevant section of audio
    y_section = y[sample_start:sample_end]
    
    # Apply bandpass filter (150Hz-1800Hz)
    nyquist = 0.5 * sr
    low = 150 / nyquist
    high = 1800 / nyquist
    sos = butter(N=6, Wn=[low, high], btype='band', output='sos')
    y_filtered = sosfilt(sos, y_section)
    
    # Compute the envelope of the filtered signal
    y_abs = np.abs(y_filtered)
    
    # Apply smoothing to the envelope
    window_size = int(0.01 * sr)  # 10ms window
    y_envelope = np.convolve(y_abs, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks in the envelope with customizable parameters
    peaks, _ = find_peaks(y_envelope, 
                        height=peak_height, 
                        distance=int(peak_distance * sr),  
                        prominence=peak_prominence) 
    
    # Convert peak positions to time (seconds), adding the start time offset
    hit_times = (peaks / sr) + time_start
    
    # Convert times to frame numbers
    hit_frames = [int(round(time * fps)) for time in hit_times]

    return hit_frames


def refine_audio(ball_shots_frames_audio, fps, audio_file):
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

    # Add all audio hits (since these are very consistent)
    for i in ball_shots_frames_audio:
        ball_shots_frames_final.add(i)


    ## GAP CHECKING 

    # Next, we need to take into account that there will be more silent hits that our audio model might not recognize, 
    # Therefore, we check if we have large gaps in the audio results and then refine the audio detection by detecting
    # lower peaks in the audio signal


    thresh = fps * 1.5 # 1.5 seconds

    for idx in range(0, len(ball_shots_frames_audio) - 1, 1):
        if ball_shots_frames_audio[idx + 1] - ball_shots_frames_audio[idx] > thresh:
            
            peak_heights = [0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.009,0.008]

            for peak_height in peak_heights:
                refined_hits = get_ball_shot_frames_audio_refinement(audio_file, fps, frames_start= ball_shots_frames_audio[idx], 
                                                                      frames_end = ball_shots_frames_audio[idx + 1], peak_height = peak_height)
            
                if refined_hits:
                    for i in refined_hits:
                        # Too close (i.e. still from old racket hit signal)
                        if (ball_shots_frames_audio[idx]  <= i <= ball_shots_frames_audio[idx] + fps//10) or (ball_shots_frames_audio[idx + 1] - fps//10  <= i <= ball_shots_frames_audio[idx + 1]):
                            pass
                        else:
                            ball_shots_frames_final.add(i)
                    break 
            

                   

    ball_shots_frames_final = sorted(list(ball_shots_frames_final))
    return ball_shots_frames_final





def draw_ball_hits(video_frames, hit_frames):
    """ Draw the ball hits on the video frames """
    output_video_frames = []
    counter = 0
    for i,frame in enumerate(video_frames):
        cv2.putText(frame, f"Racket Hits: {counter}", (10, 100), cv2.FONT_HERSHEY_DUPLEX , 1, (0,255,0), 2)
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