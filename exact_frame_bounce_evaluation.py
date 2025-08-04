from utils import (read_video, 
                   save_video, 
                   write_track, 
                   get_ball_shot_frames_audio, 
                   draw_racket_hits, 
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   create_black_video,
                   scraping_data_for_inference,
                   detect_frames_TRACKNET,
                   cluster_series,
                   filter_ball_detections_by_player,
                   draw_debug_window,
                   draw_frames_number,
                   draw_ball_landings,
                   map_court_position_to_player_id,
                   get_ball_shot_frames_visual, 
                   combine_audio_visual,
                   convert_mp4_to_mp3
                   )

from trackers import (PlayerTracker, BallTrackerNetTRACE, BallTracker)
from mini_court import MiniCourt
from ball_landing import (BounceCNN, make_prediction, evaluation_transform)
from court_line_detector import CourtLineDetector

import torch 
import pandas as pd
import info
import os 
import pickle
import numpy as np
from copy import deepcopy
import re
from collections import defaultdict



def extract_and_group_frame_numbers(directory_path):
    """
    Extracts frame numbers from filenames and groups them by video number.
    
    Args:
        directory_path (str): Path to the directory containing the image files
        
    Returns:
        list: List of lists, where each sublist contains frame numbers for a specific video
        
    Example:
        If files are: 1039_frame_59.jpg, 1039_frame_60.jpg, 1040_frame_1.jpg
        Returns: [[59, 60], [1]]
    """
    # Dictionary to group frame numbers by video number
    video_groups = defaultdict(list)
    
    # Get all files in the directory
    try:
        files = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Directory {directory_path} not found")
        return []
    
    # Pattern to match filename format: video_number_frame_frame_number.jpg
    pattern = r'^(\d+)_frame_(\d+)\.jpg$'
    
    for filename in files:
        match = re.match(pattern, filename)
        if match:
            video_number = int(match.group(1))
            frame_number = int(match.group(2))
            video_groups[video_number].append(frame_number)
    
    # Sort frame numbers within each video group
    for video_num in video_groups:
        video_groups[video_num].sort()
    
    # Convert to list of lists, sorted by video number
    result = []
    for video_num in sorted(video_groups.keys()):
        result.append(video_groups[video_num])
    
    return result






def get_ball_bounce_frames():


    y_true_bounces =  [[35, 65, 79, 106], [28, 61, 92, 127, 152, 182, 214, 240, 277, 308, 338, 363, 400], [34, 62], [42, 67, 96], [28, 57, 87]]

    


    
    all_ball_bounce_frames_predicted = []
    counter = 0
    video_numbers = [i for i in range(1039,1041)] + [i for i in range(1042,1045)]  #[i for i in range(1000,1020)] + [i for i in range(1021,1024)] + [i for i in range(1025,1027)] + [i for i in range(1028,1031)] + [i for i in range(1032,1041)] + [i for i in range(1042,1045)]
     
    for VIDEO_NUMBER_CURR in video_numbers:
       ######## CONFIG ########
        TRACE = 6 # Set trace to same amount that BounceCNN was trained on (currently : 6)
        
        # Select the player
        SELECTED_PLAYER = 'Lower' # 'Upper' or 'Lower'
        
        # Draw Options
        DRAW_MINI_COURT = True
        DRAW_STATS_BOX = True

        # Debugging Mode
        DEBUG = True

        # Video Number to run inference on
        VIDEO_NUMBER = VIDEO_NUMBER_CURR
        print(f"Running inference on video {VIDEO_NUMBER}")
        
        # Insert ground truth values for the racket hits and ball landings for best accuracy
        GT_BOUNCES_FRAMES = []
        GT_RACKET_HITS_FRAMES = []

        # Video Paths
        INPUT_VIDEO_PATH = f'data/new_input_videos/input_video_{VIDEO_NUMBER}.mp4'  #
        INPUT_VIDEO_PATH_AUDIO =  f'data/new_input_videos/input_video_{VIDEO_NUMBER}_audio.mp3'
        OUTPUT_VIDEO_PATH = f'output/final/output_video{VIDEO_NUMBER}.mp4'

        # Check if we already processed that video by looking if output with video number reference exists (for faster testing)
        if os.path.exists(OUTPUT_VIDEO_PATH):
            READ_STUBS = True
        else:
            READ_STUBS = True
        
        if not os.path.exists(INPUT_VIDEO_PATH_AUDIO):
            print(f"Converting video {VIDEO_NUMBER} to audio")
            convert_mp4_to_mp3(INPUT_VIDEO_PATH, INPUT_VIDEO_PATH_AUDIO)
            
        # Check if GPU is available
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using DEVICE: {DEVICE}")



        ######## DETECTIONS ########

        # Initialize Ball Tracker
        
        # YOLO
        ball_tracker_yolo = BallTracker(model_path = 'models/yolo11best.pt')
        
        # TrackNet
        ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
        saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location=DEVICE)
        ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
        ball_tracker_TRACKNET.to(DEVICE)
        ball_tracker_TRACKNET.eval() 

        # Initialize Player Tracker
        player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

        # Initialize Courline Detector
        courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth', machine = DEVICE) 

        # Read in the video
        video_frames, fps, video_width, video_height = read_video(INPUT_VIDEO_PATH)

        # Detect & Track Players
        player_detections = player_tracker.detect_frames(video_frames,
                                                        read_from_stub = READ_STUBS,
                                                        stub_path = f'tracker_stubs/player_detections_{VIDEO_NUMBER}.pkl')

        # Detect & Track Ball (TrackNet)
        ball_detections, ball_detections_tracknet = detect_frames_TRACKNET(video_frames, video_number=VIDEO_NUMBER, tracker=ball_tracker_TRACKNET,
                            video_width=video_width, video_height= video_height, read_from_stub = READ_STUBS, 
                            stub_path=  f'tracker_stubs/tracknet_ball_detections_{VIDEO_NUMBER}.pkl')
        
        # Detect & Track Ball (YOLO)
        ball_detections_YOLO = ball_tracker_yolo.detect_frames(video_frames, 
                                                            read_from_stub = READ_STUBS, 
                                                                stub_path = f'tracker_stubs/ball_detections_YOLO_{VIDEO_NUMBER}.pkl')
                
        ball_detections_YOLO = ball_tracker_yolo.interpolate_ball_positions(ball_detections_YOLO)

        # Detect court lines (on just the first frame, then they are fixed) 
        refined_keypoints = courtline_detector.predict(video_frames[0])

        # Filter players (such that only the two actual players are tracked)
        player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(refined_keypoints, player_detections)

        # Map player positions to player IDs
        player_position_to_id = map_court_position_to_player_id(refined_keypoints, player_detections)
        print(f"Player mapping: Upper is player_{player_position_to_id.get('Upper')}, Lower is player_{player_position_to_id.get('Lower')}")
        print(f"Selected Player : {SELECTED_PLAYER}")
        print(f"Selected Player ID : {player_position_to_id.get(SELECTED_PLAYER)}")

        # Initialize Mini Court
        mini_court = MiniCourt(video_frames[0])

        # Convert player positions to mini court positions
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                                ball_detections_YOLO,
                                                                                                                                refined_keypoints,
                                                                                                                                chosen_players_ids)
        
        mini_court_keypoints = mini_court.drawing_key_points

        # Get Racket Hits based on audio & visual information

        # Get first hit with less sensitive audio peak detection
        first_hit = (get_ball_shot_frames_audio(INPUT_VIDEO_PATH_AUDIO, fps, height = 0.01, prominence=0.01))[0]
        ball_shots_frames_visual = get_ball_shot_frames_visual(ball_detections_YOLO, fps, mode = 'yolo')
        ball_shots_frames_audio = get_ball_shot_frames_audio(INPUT_VIDEO_PATH_AUDIO, fps, plot = True)

        ball_shots_frames = combine_audio_visual(ball_shots_frames_visual= ball_shots_frames_visual,
                                                    ball_shots_frames_audio= ball_shots_frames_audio, 
                                                    fps = fps,
                                                    player_boxes = player_mini_court_detections, 
                                                    keypoints = mini_court_keypoints,
                                                    ball_detections = ball_mini_court_detections,
                                                    max_distance_param = 7,
                                                    MINI_COURT= True,
                                                    CLUSTERING= False)


        if ball_shots_frames[0] != first_hit:
            ball_shots_frames.insert(0, first_hit)
            


        print("Ball Shots from Visual : ", ball_shots_frames_visual)
        print("Ball Shots from Audio : ", ball_shots_frames_audio)
        print("Combined :", ball_shots_frames)


        ball_shots_frames_stats = ball_shots_frames.copy()

        # First, create a completely black video with same dimensions & fps of actual video 
        frame_count = len(video_frames)
        input_video_black_path = f"data/trajectory_model_videos/output_video{VIDEO_NUMBER}.mp4"
        create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)

        # Read in this video
        video_frames_black, fps, video_width, video_height = read_video(input_video_black_path)

        # Draw Ball Detection into black video
        output_frames_black = write_track(video_frames_black, ball_detections_tracknet, ball_shots_frames, trace = 10, draw_mode= 'circle')

        # Draw Keypoints (and lines) of the court into black video 
    # output_frames_black = courtline_detector.draw_keypoints_on_video(output_frames_black, refined_keypoints)

        scraping_data_for_inference(video_n= VIDEO_NUMBER, output_path = 'data_inference', input_frames = output_frames_black,
                                    ball_shots_frames = ball_shots_frames , trace = TRACE, ball_detections = ball_detections_tracknet)

        # Instantiate Bounce Model
        model_bounce = BounceCNN()
        with open('data_bounce_stubs/data_mean_std.pkl', 'rb') as f:
            data_mean_and_std = pickle.load(f)
        
        mean = data_mean_and_std[0]
        std = data_mean_and_std[1]

        # Make Predictions
        predictions, confidences, img_idxs = make_prediction(model = model_bounce, best_model_path = 'models/best_bounce_model.pth',
                                            input_frames_directory = 'data_inference/images_inference', transform = evaluation_transform(mean, std), device = DEVICE)
        
        # Get Bounce Frames
        mask = np.array(predictions) == 1
        img_idxs_bounce = np.array(img_idxs)[mask].tolist()
        ball_landing_frames = cluster_series(img_idxs_bounce, min_samples = 2, delay = 2)
        print(f"Predicted V Shaped Frames : {img_idxs_bounce}")
        print(f"Predicted Bounce Frames : {ball_landing_frames}")
        print(f"True Bounce Frames : {y_true_bounces[counter]}")


        all_ball_bounce_frames_predicted.append(ball_landing_frames)

        counter += 1 
    

    return all_ball_bounce_frames_predicted



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def complete_bounce_evaluation(y_true_bounces, y_true_non_bounces, y_pred_bounces, tolerance=3):
    """
    Complete bounce detection evaluation with simple, correct logic.
    
    Parameters:
    -----------
    y_true_bounces : list of lists
        Ground truth bounces. First element of each sublist will be ignored.
        Each sublist corresponds to one video.
    y_true_non_bounces : list of lists
        Ground truth non-bounces. Each sublist corresponds to one video.
    y_pred_bounces : list of lists  
        Predicted bounces. Each sublist corresponds to one video.
    tolerance : int, default=3
        Frame tolerance for matching bounces (±tolerance frames)
    
    Returns:
    --------
    dict : Dictionary containing complete confusion matrix and metrics
    """
    
    # Ensure all lists have the same length (same number of videos)
    num_videos = len(y_true_bounces)
    if len(y_true_non_bounces) != num_videos or len(y_pred_bounces) != num_videos:
        raise ValueError("All input lists must have the same number of sublists (videos)")
    
    print(f"Evaluating {num_videos} videos...")
    
    # Check for overlapping frames between true bounces and true non-bounces
    print("\nChecking for overlaps between true bounces and true non-bounces...")
    for video_idx in range(num_videos):
        true_bounces_video = y_true_bounces[video_idx]
        true_non_bounces_video = y_true_non_bounces[video_idx]
        
        # Remove first element from true bounces for this video
        if len(true_bounces_video) > 1:
            true_bounces_modified = true_bounces_video[1:]  # Remove first element
        else:
            true_bounces_modified = []


    # Initialize counters
    tp = 0
    fp = 0
    
    # Process each video separately
    for video_idx in range(num_videos):
        # Get data for this video
        true_bounces_video = y_true_bounces[video_idx]
        true_non_bounces_video = y_true_non_bounces[video_idx]
        pred_bounces_video = y_pred_bounces[video_idx]
        
        # Remove first element from true bounces for this video
        if len(true_bounces_video) > 1:
            true_bounces_modified = true_bounces_video[1:]  # Remove first element
        else:
            true_bounces_modified = []  # No bounces to evaluate for this video
        
        print(f"Video {video_idx}: True bounces: {len(true_bounces_modified)}, "
              f"True non-bounces: {len(true_non_bounces_video)}, "
              f"Predicted bounces: {len(pred_bounces_video)}")
        
        # 1. TRUE POSITIVES: Count matches between predicted and true bounces (WITH tolerance)
        tp_video = 0
        tp_pred_frames = set()  # Track which predictions were already counted as TP
        for pred_frame in pred_bounces_video:
            for true_frame in true_bounces_modified:
                if abs(pred_frame - true_frame) <= tolerance:
                    tp_video += 1
                    tp_pred_frames.add(pred_frame)  # Mark this prediction as TP
                    break  # Count each prediction only once
        
        # 2. FALSE POSITIVES: Count matches between predicted and true non-bounces (WITHOUT tolerance)
        # BUT exclude predictions that were already counted as True Positives (since we use delay in our cluster_series(), its possible that when
        # the predicted bounce was found shortly before the true bounce (which can happen due to the delay in cluster_series(), say it already finds
        # a bounce at k+1 , where k is the bounce frame (since it already sees some v pattern), then with the delay of 2 in cluster_series(), it would
        # choose a non-bounce frame as bounce, even though it actually detected the correct bounce.) )
        fp_video = 0
        for pred_frame in pred_bounces_video:
            if pred_frame not in tp_pred_frames and pred_frame in true_non_bounces_video:
                fp_video += 1
        
        # Add to totals
        tp += tp_video
        fp += fp_video
        
        print(f"  TP: {tp_video}, FP: {fp_video}")
    
    # 3. FALSE NEGATIVES: Total true bounces minus true positives
    total_true_bounces = sum(len(video[1:]) if len(video) > 1 else 0 for video in y_true_bounces)
    fn = total_true_bounces - tp
    
    # 4. TRUE NEGATIVES: Total true non-bounces minus false positives
    total_true_non_bounces = sum(len(video) for video in y_true_non_bounces)
    tn = total_true_non_bounces - fp
    
   

    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    

    print(f"\nComplete Bounce Detection Evaluation Results (±{tolerance} frame tolerance for bounces):")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall : {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Create confusion matrix array
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Create labels with counts and percentages
    total = tp + tn + fp + fn
    labels = np.array([
        [f'{tn}', f'{fp}'], 
        [f'{fn}', f'{tp}']
    ])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=['Predicted: No Bounce', 'Predicted: Bounce'],
                yticklabels=['Actual: No Bounce', 'Actual: Bounce'])
    
    plt.title(f'Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    
    


if __name__ == "__main__":

    # The test set true bounces (exact frames)
    y_true_bounces =  [[35, 79, 106], [28, 61, 92, 127, 152, 182, 214, 240, 277, 308, 338, 363, 400], [34, 62], [42, 67, 96], [28, 57, 87]]
    
    # The test set predicted bounces (exact frames)
    y_pred_bounces = get_ball_bounce_frames()

    # The test set true non-bounces (exact frames)
    y_true_non_bounces = extract_and_group_frame_numbers('data/trajectory_model_dataset/circles/test/no_bounce')

    # Frame-level evaluation (based on some tolerance level)
    results = complete_bounce_evaluation(y_true_bounces, y_true_non_bounces, y_pred_bounces, tolerance=2)
    # at tol = 3, it misses 2 bounces out of 20 total bounces, 10% error rate
    # at tol = 3, it has 2 extra bounces that do not exist out of 20 total bounces, 10% error rate



   # y_true_bounces =  [[76],[56,92],[23,87], [20,49,76,101,141],[36,67,101],
   #                                 [18,86], [14,48,80], [11,51,80,119,150,180,211,238,271,312], [16,48,79,109,139,172,201],[3, 37,68],
   #                                 [23, 54, 84, 127, 152, 183, 213, 243, 267, 304, 335, 384, 417, 444, 478, 504,530, 565, 592, 636, 664, 710,739, 768,805,835],
   #                                 [19, 58], [34, 65, 91], [34, 63, 97, 129], [26, 65, 94], [28, 56, 83, 121, 149, 182, 214, 241, 268, 296, 335, 364],
   #                                 [17, 49], [30, 60], [11, 46, 77], [26, 61], [149, 177, 206], [32, 65, 91, 130], [33, 59, 90, 124, 156, 186, 222, 252, 285, 313],
   #                                 [17, 57, 85, 117, 142, 185, 213, 245, 271, 303], [19, 51, 95, 128], [30, 66, 108], [36, 69, 96, 133, 164, 196, 229],
   #                                 [32,75], [29,56, 86, 111,174,220], [36,71,104,131,167,195], [31,67,99,137,170,199,235,270,298,324,357,384,428,472,521,552],
   #                                 [29,74,99,129], [29,65], [33,65,95,131,163,191], [32,67,96,143,170,203,230,266,291,324,352,399,428], [35, 65, 79, 106],
   #                                 [28, 61, 92, 127, 152, 182, 214, 240, 277, 308, 338, 363, 400], [34, 62], [42, 67, 96], [28, 57, 87]
   #                                 ]