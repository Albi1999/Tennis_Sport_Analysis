from utils import (read_video, 
                   save_video, 
                   refine_audio, 
                   write_track, 
                   get_ball_shot_frames_audio, 
                   draw_racket_hits, 
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   create_black_video,
                   scraping_data_for_inference,
                   detect_frames_TRACKNET,
                   create_player_stats_box_video,
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
import cv2 
import torch 
from copy import deepcopy
import pandas as pd
import info
import os 
import pickle
import numpy as np

# TODO: Clean the code and remove the unnecessary parts

def main():


    ######## CONFIG ########
    TRACE = 6 # Set trace to same amount that BounceCNN was trained on (currently : 6)
    
    # Select the player
    SELECTED_PLAYER = 'Lower' # 'Upper' or 'Lower'
    
    # Draw Options
    DRAW_MINI_COURT = True
    DRAW_STATS_BOX = True

    # Debugging Mode
    DEBUG = True

    # Video to run inference on
    # Note : for video 116, change in mini_court.convert_bounding_boxes_to_mini_court_coordinates(...) ball_detections_YOLO to ball_detections
    # (leads to better results)
    video_number = 101
    print(f"Running inference on video {video_number}")
    
    # Insert ground truth values for the racket hits and ball landings for best accuracy
    ground_truth_bounce = []
    ground_truth_racket_hits = []

    # Video Paths
    input_video_path = f'data/input_video{video_number}.mp4'  #
    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
    output_video_path = f'output/final/output_video{video_number}.mp4'

    # Check if we already processed that video by looking if output with video number reference exists (for faster testing)
    if os.path.exists(output_video_path):
        READ_STUBS = True
    else:
        READ_STUBS = False
    
    if not os.path.exists(input_video_path_audio):
        print(f"Converting video {video_number} to audio")
        convert_mp4_to_mp3(input_video_path, input_video_path_audio)
        
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")



    ######## DETECTIONS ########

    # Initialize Ball Tracker
    # YOLO
    ball_tracker_yolo = BallTracker(model_path = 'models/yolo11best.pt')
    # TrackNet
    ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
    saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location=device)
    ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
    ball_tracker_TRACKNET.to(device)
    ball_tracker_TRACKNET.eval() 

    # Initialize Player Tracker
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

    # Initialize Courline Detector
    courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth', machine = device) 

    # Read in the video
    video_frames, fps, video_width, video_height = read_video(input_video_path)

    # Detect & Track Players
    player_detections = player_tracker.detect_frames(video_frames,
                                                    read_from_stub = READ_STUBS,
                                                    stub_path = f'tracker_stubs/player_detections_{video_number}.pkl')

    # Detect & Track Ball (TrackNet)
    ball_detections, ball_detections_tracknet = detect_frames_TRACKNET(video_frames, video_number = video_number, tracker =ball_tracker_TRACKNET,
                        video_width=video_width, video_height= video_height, read_from_stub = READ_STUBS, 
                        stub_path=  f'tracker_stubs/tracknet_ball_detections_{video_number}.pkl')
    
    # Detect & Track Ball (YOLO)
    ball_detections_YOLO = ball_tracker_yolo.detect_frames(video_frames, 
                                                           read_from_stub = READ_STUBS, 
                                                            stub_path = f'tracker_stubs/ball_detections_YOLO_{video_number}.pkl')
            
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
    first_hit = (get_ball_shot_frames_audio(input_video_path_audio, fps, height = 0.01, prominence=0.01))[0]
    ball_shots_frames_visual = get_ball_shot_frames_visual(ball_detections_YOLO, fps, mode = 'yolo')
    ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)

    ball_shots_frames = combine_audio_visual(ball_shots_frames_visual= ball_shots_frames_visual,
                                                  ball_shots_frames_audio= ball_shots_frames_audio, 
                                                  fps = fps,
                                                  player_boxes = player_mini_court_detections, 
                                                  keypoints = mini_court_keypoints,
                                                  ball_detections = ball_mini_court_detections,
                                                  max_distance_param = 7,
                                                  adjustment = 0,
                                                  MINI_COURT= True,
                                                  CLUSTERING= False)


    if ball_shots_frames[0] != first_hit:
        ball_shots_frames.insert(0, first_hit)
        


    print("Ball Shots from Visual : ", ball_shots_frames_visual)
    print("Ball Shots from Audio : ", ball_shots_frames_audio)
    print("Combined :", ball_shots_frames)

  #  ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)
  #  ball_shots_frames = refine_audio(ball_shots_frames_audio, fps, input_video_path_audio)
  #  print("Ball Shots from Audio : ", ball_shots_frames_audio)
  #  print("Audio Refinment :", ball_shots_frames)
    ball_shots_frames_stats = ball_shots_frames.copy()

    # First, create a completely black video with same dimensions & fps of actual video 
    frame_count = len(video_frames)
    input_video_black_path = f"data/trajectory_model_videos/output_video{video_number}.mp4"
    create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)

    # Read in this video
    video_frames_black, fps, video_width, video_height = read_video(input_video_black_path)

    # Draw Ball Detection into black video
    output_frames_black = write_track(video_frames_black, ball_detections_tracknet, ball_shots_frames, trace = 10, draw_mode= 'circle')

    # Draw Keypoints (and lines) of the court into black video 
   # output_frames_black = courtline_detector.draw_keypoints_on_video(output_frames_black, refined_keypoints)

    scraping_data_for_inference(video_n= video_number, output_path = 'data_inference', input_frames = output_frames_black,
                                 ball_shots_frames = ball_shots_frames , trace = TRACE, ball_detections = ball_detections_tracknet)

    # Instantiate Bounce Model
    model_bounce = BounceCNN()
    with open('data_bounce_stubs/data_mean_std.pkl', 'rb') as f:
        data_mean_and_std = pickle.load(f)
    
    mean = data_mean_and_std[0]
    std = data_mean_and_std[1]

    # Make Predictions
    predictions, confidences, img_idxs = make_prediction(model = model_bounce, best_model_path = 'models/best_bounce_model.pth',
                                         input_frames_directory = 'data_inference/images_inference', transform = evaluation_transform(mean, std), device = device)
    
    # Get Bounce Frames
    mask = np.array(predictions) == 1
    img_idxs_bounce = np.array(img_idxs)[mask].tolist()
    ball_landing_frames = cluster_series(img_idxs_bounce)
    print(f"Predicted V Shaped Frames : {img_idxs_bounce}")
    print(f"Predicted Bounce Frames : {ball_landing_frames}")
    
    # TODO: Create a function to select ONLY the ball landing frames in the upper part of the court
    upper_court_balls_frames, lower_court_balls_frames = filter_ball_detections_by_player(ball_landing_frames, ball_mini_court_detections, mini_court)
    print(f"Upper Court Bounce Frames : {upper_court_balls_frames}")
    print(f"Lower Court Bounce Frames : {lower_court_balls_frames}")
    if SELECTED_PLAYER == 'Upper':
        player_balls_frames = lower_court_balls_frames
    else:
        player_balls_frames = upper_court_balls_frames
    print(f"Selected player is {SELECTED_PLAYER} so the bounce used in ball heatmap are in the opposite court and are: {player_balls_frames}")
    
    


    ######## GROUND TRUTH ########
    
    # TODO: Remove the too bad videos from this statement (?)
    if DEBUG:
        # Fixing some mistakes in the ball shots frames

        if video_number == 101:
            ground_truth_bounce = [20,50,77,106,138,168,197,230,270,301]
            ball_shots_frames_stats.remove(123)
            ball_shots_frames_stats.remove(115)
            ball_shots_frames_stats.append(118)

        if video_number == 102:
            ground_truth_bounce = [13,41,73,105,131,159,190,221,270,299,329,363,414]

        if video_number == 103:
            ground_truth_bounce = [23,66,97]

        if video_number == 105:
            ground_truth_bounce = [48,None,115,146,208,268]
            ball_shots_frames_stats.append(186)

        if video_number == 107:
            ground_truth_bounce = [69,107,133,172,202,279]
            ball_shots_frames_stats.remove(269)
            ball_shots_frames_stats.remove(203)

        if video_number == 108:
            ground_truth_bounce = [29, 78, 104]

        if video_number == 109:
            ground_truth_bounce = [30, 55, 86, 111, 144, 169, 203, 244]
            ball_shots_frames_stats.remove(47)
            ball_shots_frames_stats.remove(61)
            ball_shots_frames_stats.append(65)
            ball_shots_frames_stats.remove(116)
            ball_shots_frames_stats.append(124)
            ball_shots_frames_stats.remove(148)
            ball_shots_frames_stats.append(153)
            ball_shots_frames_stats.remove(171)
            ball_shots_frames_stats.append(177)
            ball_shots_frames_stats.append(211)

        if video_number == 110:
            ground_truth_bounce = [23, 60, 94, 123, 162, 200, 244, 273, 305, 334, 363, 390]
            ball_shots_frames_stats.remove(90)
            ball_shots_frames_stats.append(104)
            ball_shots_frames_stats.remove(112)
            ball_shots_frames_stats.remove(132)
            ball_shots_frames_stats.append(136)
            ball_shots_frames_stats.remove(169)
            ball_shots_frames_stats.append(174)
            ball_shots_frames_stats.remove(203)
            ball_shots_frames_stats.append(213)
            ball_shots_frames_stats.remove(229)
            ball_shots_frames_stats.remove(243)
            ball_shots_frames_stats.append(252)
            ball_shots_frames_stats.remove(270)
            ball_shots_frames_stats.append(282)
            ball_shots_frames_stats.remove(299)
            ball_shots_frames_stats.append(310)
            ball_shots_frames_stats.remove(327)
            ball_shots_frames_stats.append(340)
            ball_shots_frames_stats.remove(359)
            ball_shots_frames_stats.append(372)
            ball_shots_frames_stats.remove(385)
            ball_shots_frames_stats.append(400)
            ball_shots_frames_stats.remove(414)
        
        if video_number == 111:
            ground_truth_bounce = [43, 76, 114, 144, 171, 207, 236]
        
        if video_number == 112:
            ground_truth_bounce = [45, 83, 109, 150, 178, 224, 268, 298, 330, 363, 405, 438]
            ball_shots_frames_stats.remove(100)
            ball_shots_frames_stats.append(158)
            ball_shots_frames_stats.remove(167)
            ball_shots_frames_stats.append(193)
            ball_shots_frames_stats.remove(317)
            ball_shots_frames_stats.append(445)
            ball_shots_frames_stats.remove(466)
        
        if video_number == 113:
            ground_truth_bounce = [26, 67, 96, 133, 148]
            ball_shots_frames_stats.append(33)
            ball_shots_frames_stats.append(76)
            ball_shots_frames_stats.remove(144)
        
        if video_number == 114:
            ground_truth_bounce = [19, 58, 83, 123, 152, 228]
            ball_shots_frames_stats.remove(65)
            ball_shots_frames_stats.append(69)
            ball_shots_frames_stats.remove(153)
            ball_shots_frames_stats.remove(219)
            ball_shots_frames_stats.remove(249)

        if video_number == 115:
            ground_truth_bounce = [17, 50, 75, 120, 150, 183, 213, 242]
            ball_shots_frames_stats.remove(44)
            ball_shots_frames_stats.append(156)
            ball_shots_frames_stats.remove(169)
        
        if video_number == 116:
            ground_truth_bounce = [12, 51, 81]
            ball_shots_frames_stats.remove(95)
        
        if video_number == 117:
            ground_truth_bounce = [14, 45, 74, 104, 138, 165, 196]
            ball_shots_frames_stats.remove(98)
        
        if video_number == 118:
            ground_truth_bounce = [38, 69, 95, 145, 171, 208, 237, 281, 309]
            ball_shots_frames_stats.append(26)
            ball_shots_frames_stats.append(45)
            ball_shots_frames_stats.append(74)
            ball_shots_frames_stats.remove(95)
            ball_shots_frames_stats.append(109)
            ball_shots_frames_stats.remove(225)
            ball_shots_frames_stats.remove(321)
            
        if video_number == 1000:
            ground_truth_bounce = [75]
        
        if video_number == 1001:
            ground_truth_bounce = [56, 92]
        
        if video_number == 1002:
            ground_truth_bounce = [22, 51, 86]
            
        if video_number == 1003:
            ground_truth_bounce = [20, 50, 76, 101, 141]
            
        ball_landing_frames_stats = ground_truth_bounce.copy()
        ball_shots_frames_stats = sorted(ball_shots_frames_stats)
 
        # Get the racket hit frames for the player
        ball_shots_frames_upper, ball_shots_frames_lower = filter_ball_detections_by_player(ball_shots_frames_stats, ball_mini_court_detections, mini_court)
        print(f"Ball Shots Upper Player: {ball_shots_frames_upper}")
        print(f"Ball Shots Lower Player: {ball_shots_frames_lower}")
        if ball_shots_frames_upper[0] < ball_shots_frames_lower[0]:
            serve_player = 'Upper'
        else:
            serve_player = 'Lower'
        print(f"The Player who serves is: {serve_player}")

        # Print the ground truth values for the racket hits and ball landings for debugging purposes
        if len(ball_shots_frames_stats) > 0 and len(ground_truth_bounce) > 0:
            print(f"Ground Truth Racket Hit frames : {ball_shots_frames_stats}")
            print(f"Ground Truth Bounce Frames : {ground_truth_bounce}")

    ######## MATCH STATS ########

    player_stats_data = [{
        'frame_num': 0,

        # player 1 stats
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_min_shot_speed': float('inf'),
        'player_1_max_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_1_hits_counter': 0,
        'player_1_distance_covered': 0,
        'player_1_score_probability': 0,

        #player 2 stats
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_min_shot_speed': float('inf'),
        'player_2_max_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
        'player_2_hits_counter': 0,
        'player_2_distance_covered': 0,
        'player_2_score_probability': 0,
    } ]

    # Track player positions for calculating total distance
    player_positions_history = {1: [], 2: []}
    
    # Compute score prediction and probability for each ball shot
    # score_prediction, score_probability = mini_court.predict_score(ball_mini_court_detections, ball_landing_frames, player_balls_frames, player_position_to_id, ball_shots_frames_stats)
    
    # Loop over all ball shots except the last one since it doesn't have an answer shot.
    for ball_shot_ind in range(len(ball_shots_frames_stats)):
        start_frame = ball_shots_frames_stats[ball_shot_ind]               # Starting frame of the ball shot
        end_frame = ball_landing_frames_stats[ball_shot_ind]               # Ending frame of the ball shot
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps        # Time taken by the ball to travel from the player to the opponent

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = euclidean_distance(ball_mini_court_detections[start_frame][1],
                                                             ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels, 
                                                                           info.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()) # Distance covered by the ball in meters

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6  # 3.6 to convert m/s to km/h

        # Player who shot the ball
        player_position = player_mini_court_detections[start_frame] # Player positions at the start of the ball shot
        player_shot_ball = min(player_position.keys(), key = lambda player_id: euclidean_distance(player_position[player_id], 
                                                                                                    ball_mini_court_detections[start_frame][1]))
        
        # Opponent player's speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2  # Opponent player's ID

        # Track player distances between frames
        for player_id in [1, 2]:
            # Calculate total distance covered by this player between shots
            player_distance = 0
            for frame in range(start_frame, end_frame):
                if (player_id in player_mini_court_detections[frame] and 
                    player_id in player_mini_court_detections[frame+1]):
                    
                    dist_pixels = euclidean_distance(
                        player_mini_court_detections[frame][player_id],
                        player_mini_court_detections[frame+1][player_id]
                    )
                    dist_meters = convert_pixel_distance_to_meters(
                        dist_pixels,
                        info.DOUBLE_LINE_WIDTH,
                        mini_court.get_width_of_mini_court()
                    )
                    player_distance += dist_meters

            player_positions_history[player_id].append(player_distance)

        # Check if opponent player was detected in both frames
        if opponent_player_id in player_mini_court_detections[start_frame] and opponent_player_id in player_mini_court_detections[end_frame]:
            distance_covered_by_opponent_pixels = euclidean_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                    player_mini_court_detections[end_frame][opponent_player_id]) 
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels, 
                                                                                info.DOUBLE_LINE_WIDTH,
                                                                                mini_court.get_width_of_mini_court())
            opponent_speed = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6  # 3.6 to convert m/s to km/h
        else:
            # Handle missing player detection
            distance_covered_by_opponent_meters = 0
            opponent_speed = 0

        # Calculate score probability based on shot speed and opponent position
        # (Simple model: faster shots with opponent far from ball landing position have higher probability)
        shot_probability = min(95, 40 + (speed_of_ball_shot / 3))  # Base probability on shot speed
        # TODO: Use the correct shot probability once it's ready

        # Update player stats
        current_player_stats = deepcopy(player_stats_data[-1]) # Copy of previous stats
        current_player_stats['frame_num'] = start_frame
        
        # Player who shot the ball stats
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        
        if start_frame in ball_shots_frames_upper:
            # This shot was made by the upper player
            upper_player_id = player_position_to_id.get('Upper')
            current_player_stats[f'player_{upper_player_id}_hits_counter'] += 1
        elif start_frame in ball_shots_frames_lower:
            # This shot was made by the lower player
            lower_player_id = player_position_to_id.get('Lower')
            current_player_stats[f'player_{lower_player_id}_hits_counter'] += 1
            
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_score_probability'] = shot_probability
        
        # Update min and max shot speeds
        # Min
        current_player_stats[f'player_{player_shot_ball}_min_shot_speed'] = min(
            current_player_stats[f'player_{player_shot_ball}_min_shot_speed'], 
            speed_of_ball_shot
        )
        # Max
        current_player_stats[f'player_{player_shot_ball}_max_shot_speed'] = max(
            current_player_stats[f'player_{player_shot_ball}_max_shot_speed'], 
            speed_of_ball_shot
        )
        
        # Update distance covered
        if player_positions_history[player_shot_ball]:
            player_distance = player_positions_history[player_shot_ball][-1]
            current_player_stats[f'player_{player_shot_ball}_distance_covered'] += player_distance

        # Opponent player stats
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += opponent_speed
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = opponent_speed
        
        # Update opponent distance covered
        if player_positions_history[opponent_player_id]:
            opponent_distance = player_positions_history[opponent_player_id][-1]
            current_player_stats[f'player_{opponent_player_id}_distance_covered'] += opponent_distance

        player_stats_data.append(current_player_stats)
    
    # Convert player stats data to a DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num':list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left') # Merge the player stats data with the frames data
    player_stats_data_df = player_stats_data_df.ffill() # Fill the missing values by replacing them with the previous value

    # Calculate average shot speed for each player
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']

    # Calculate average player speed for each player
    # We calculate the average player speed by dividing the total player speed by the number of shots of the opponent player
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']




    ######## ANIMATIONS ########
    '''
    # Create player heatmap
    player_heatmap_path = mini_court.create_player_heatmap_animation(
        player_mini_court_detections,
        output_path=f"output/animations/player_heatmap_animation{video_number}.mp4",
        mask_path=f"output/masks/player_heatmap_mask{video_number}.npy")
    print(f"Player Heatmap animation saved to: {player_heatmap_path}")

    # Create ball heatmap
    ball_heatmap_path = mini_court.create_ball_heatmap_animation(
        player_mini_court_detections,
        ball_mini_court_detections,
        ball_landing_frames,
        player_balls_frames,
        output_path=f"output/animations/ball_heatmap_animation{video_number}.mp4",
        sigma=15,
        color_map=cv2.COLORMAP_HOT,
        mask_path=f"output/masks/ball_heatmap_mask{video_number}.npy")
    print(f"Ball Heatmap animation saved to: {ball_heatmap_path}")

    # Create score heatmap (combination of player and ball heatmaps)
    score_heatmap_path = mini_court.create_scoring_heatmap_animation(
        player_mini_court_detections, 
        ball_mini_court_detections, 
        ball_landing_frames,
        output_path=f"output/animations/scoring_heatmap_animation{video_number}.mp4", 
        color_map=cv2.COLORMAP_HOT)
    print(f"Score Heatmap animation saved to: {score_heatmap_path}")
    
    # Create player stats box video
    stats_box_video_path = create_player_stats_box_video(
        player_stats_data_df, video_number, fps=fps,
        selected_player=SELECTED_PLAYER,
        player_mapping=player_position_to_id)
    print(f"Player Stats Box saved to: {stats_box_video_path}")
    '''




    ######## DRAW OUTPUT ########
    
    # Draw Player Detection
    output_frames = player_tracker.draw_ellipse_bboxes(video_frames, player_detections, SELECTED_PLAYER)

    # Draw Ball Detection
    output_frames = write_track(video_frames, ball_detections_tracknet)

    # Draw keypoints, according to the first frame
    all_keypoints = courtline_detector.detect_keypoints_for_all_frames(video_frames)
    output_frames = courtline_detector.draw_keypoints_on_video_dinamically(output_frames, all_keypoints)

    # Draw Mini Court
    if DRAW_MINI_COURT:
        output_frames = mini_court.draw_mini_court(output_frames)
   
        output_frames = mini_court.draw_ball_landing_heatmap(output_frames, player_balls_frames, ball_mini_court_detections, ball_shots_frames_upper, ball_shots_frames_lower, selected_player=SELECTED_PLAYER)

        #output_frames = mini_court.draw_player_distance_heatmap(output_frames, player_mini_court_detections, selected_player=SELECTED_PLAYER)

        output_frames = mini_court.draw_shot_trajectories(output_frames, player_mini_court_detections, ball_mini_court_detections, ball_landing_frames_stats, ball_shots_frames_stats, serve_player=serve_player)     

        output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections,object_type='player')
        #output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, object_type='ball')

    # Draw player stats box
    if DRAW_STATS_BOX:
        output_frames = draw_player_stats(output_frames, player_stats_data_df, 
                                            selected_player=SELECTED_PLAYER,
                                            player_mapping=player_position_to_id)

    # Draw Frames Number, Racket Hits and Ball Landings for debugging purposes
    if DEBUG:
        output_frames = draw_debug_window(output_frames)
        output_frames = draw_frames_number(output_frames)
        output_frames = draw_racket_hits(output_frames, ball_shots_frames_stats)
        output_frames = draw_ball_landings(output_frames, ball_landing_frames, ground_truth_bounce, ball_detections_tracknet)

    # Save video
    save_video(output_frames, output_video_path, fps)
    print(f"Output video saved to: {output_video_path}")


if __name__ == '__main__':
    main()

