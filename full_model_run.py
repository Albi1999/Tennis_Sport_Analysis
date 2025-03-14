from utils import (read_video, 
                   save_video, 
                   infer_model, 
                   remove_outliers, 
                   split_track, 
                   refine_audio, 
                   interpolation, 
                   write_track, 
                   get_ball_shot_frames_audio, 
                   convert_mp4_to_mp3,
                   draw_ball_hits, 
                   convert_ball_detection_to_bbox, 
                   get_ball_shot_frames_visual,
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   remove_outliers_final,
                   create_black_video,
                   scraping_data,
                   scraping_data_for_inference,
                   splitting_data,
                   detect_frames_TRACKNET,
                   create_player_stats_box_video)
from trackers import (PlayerTracker, BallTracker, BallTrackerNetTRACE)
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
import sklearn.cluster



def main():

    DRAW_MINI_COURT = False
    SAVE_STATS_BOX_SEPARATELY = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Video to run inference on
    video_number = 100

    input_video_path = f'data/input_video{video_number}.mp4'  #

    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
    convert_mp4_to_mp3(input_video_path, input_video_path_audio)
        
    output_video_path = f'output/trajectory_model_videos/output_video{video_number}.mp4'

    # Check if we already processed that video by looking if output with video number reference exists (for faster testing)
    if os.path.exists(output_video_path):
        READ_STUBS = True
    else:
        READ_STUBS = False

    # Initialize Ball Tracker
    ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
    saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location= device)
    ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
    ball_tracker_TRACKNET.to(device)
    ball_tracker_TRACKNET.eval() 

    # Initialize Player Tracker
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

    # Initialize Courline Detector
    courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth') 

    # Read in the video
    video_frames_real, fps, video_width, video_height = read_video(input_video_path)

    # Detect & Track Players
    player_detections = player_tracker.detect_frames(video_frames_real,
                                                    read_from_stub = READ_STUBS,
                                                    stub_path = f'tracker_stubs/player_detections_{video_number}.pkl')


    # Detect & Track Ball
    ball_detections, ball_detections_tracknet = detect_frames_TRACKNET(video_frames_real, video_number = video_number, tracker =ball_tracker_TRACKNET,
                        video_width=video_width, video_height= video_height, read_from_stub = READ_STUBS, 
                        stub_path=  f'tracker_stubs/tracknet_ball_detections_{video_number}.pkl')


    # Detect court lines (on just the first frame, then they are fixed) 
    refined_keypoints = courtline_detector.predict(video_frames_real[0])


    # Filter players (such that only the two actual players are tracked)
    player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(refined_keypoints, player_detections)
            

    # Initialize Mini Court
    mini_court = MiniCourt(video_frames_real[0])


    # Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                            ball_detections,
                                                                                                                            refined_keypoints,
                                                                                                                            chosen_players_ids)

    # Get Racket Hits based on audio
    ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)
    ball_shots_frames = refine_audio(ball_shots_frames_audio, fps, input_video_path_audio)


    print("Ball Shots from Audio : ", ball_shots_frames_audio)
    print("Audio Refinment :", ball_shots_frames)


    ## Draw Output 

    # on black video (needed for bounce detection)

    # First, create a completely black video with same dimensions & fps of actual video 
    frame_count = len(video_frames_real)
    input_video_black_path = f"data/trajectory_model_videos/output_video{video_number}.mp4"
    create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)

    # Read in this video
    video_frames_black, fps, video_width, video_height = read_video(input_video_black_path)


    # Draw Ball Detection onto black video
    output_frames_black = write_track(video_frames_black, ball_detections_tracknet, ball_shots_frames, trace = 10, draw_mode= 'circle')


    # Draw Keypoints (and lines) of the court onto black video 
    output_frames_black = courtline_detector.draw_keypoints_on_video(output_frames_black, refined_keypoints)

    img_idxs = scraping_data_for_inference(video_n= video_number, output_path = 'data_inference', input_frames = output_frames_black,
                                 ball_shots_frames = ball_shots_frames , trace = 10, ball_detections = ball_detections_tracknet)

    # Instantiate Bounce Model
    model_bounce = BounceCNN()
    with open('data_bounce_stubs/data_mean_std.pkl', 'rb') as f:
        data_mean_and_std = pickle.load(f)
    
    mean = data_mean_and_std[0]
    std = data_mean_and_std[1]


    predictions, confidences = make_prediction(model = model_bounce, best_model_path = 'models/best_bounce_model.pth',
                                         input_frames_directory = 'data_inference/images_inference', transform = evaluation_transform(mean, std), device = device)
    
    mask = np.array(predictions) == 1

    img_idxs_bounce = np.array(img_idxs)[mask].tolist()



 #   predictions_bounce_frame = np.where(np.array(predictions) == 1)[0].tolist()
 #   predictions_no_bounce_frame = np.where(np.array(predictions) == 0)[0].tolist()

    # Sort by confidences
 #   bounce_frames = sorted(list(zip(predictions_bounce_frame_idx, confidences)), key = lambda x : x[1], reverse = True)
 #   no_bounce_frames = sorted(list(zip(predictions_no_bounce_frame_idx, confidences)), key = lambda x : x[1], reverse = True)

 #   print("These are the predicted bounce frames (ordered by confidences) : ")
    
 #   for i in bounce_frames:
 #       print(f"Bounce at frame : {i[0]} with confidence {i[1]}")




    ball_landing_frames = None # TODO : clustering + min + some logic based on size on img_idxs_bounce
    first_player_balls_frames = None # TODO : racket hits [::2]



    # on actual video (our final output)


    # Speed statistics

    player_stats_data = [{
        'frame_num': 0,

        # player 1 stats
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        #player 2 stats
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    } ]

    # Loop over all ball shots except the last one since it doesn't have an answer shot.
    for ball_shot_ind in range(len(ball_shots_frames)-1):
        start_frame = ball_shots_frames[ball_shot_ind]               # Starting frame of the ball shot
        end_frame = ball_shots_frames[ball_shot_ind+1]               # Ending frame of the ball shot
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps  # Time taken by the ball to travel from the player to the opponent

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
        opponent_player_id = 1 if player_shot_ball == 2 else 2 # Opponent player's ID
        distance_covered_by_opponent_pixels = euclidean_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                    player_mini_court_detections[end_frame][opponent_player_id]) 
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels, 
                                                                           info.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()) # Distance covered by the opponent player in meters

        opponent_speed = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6 # 3.6 to convert m/s to km/h

        # Update player stats
        current_player_stats = deepcopy(player_stats_data[-1]) # Copy of previous stats
        current_player_stats['frame_num'] = start_frame
        
        # Player who shot the ball stats
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        # Opponent player stats
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += opponent_speed
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = opponent_speed

        player_stats_data.append(current_player_stats) # Append the updated stats
    
    # Convert player stats data to a DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num':list(range(len(video_frames_real)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left') # Merge the player stats data with the frames data
    player_stats_data_df = player_stats_data_df.ffill() # Fill the missing values by replacing them with the previous value

    # Calculate average shot speed for each player
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']

    # Calculate average player speed for each player
    # We calculate the average player speed by dividing the total player speed by the number of shots of the opponent player
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']


    
    # Generate score heatmap by combining player and ball heatmaps
    player_heatmap_path = mini_court.create_player_heatmap_animation(
        player_mini_court_detections,
        output_path=f"output/animations/player_heatmap_animation{video_number}.mp4",
        mask_path=f"output/masks/player_heatmap_mask{video_number}.npy"
    )
    
    print(f"Player Heatmap animation saved to: {player_heatmap_path}")

    ball_heatmap_path = mini_court.create_ball_heatmap_animation(
        player_mini_court_detections,
        ball_mini_court_detections,
        ball_landing_frames,
        first_player_balls_frames,
        output_path=f"output/animations/ball_heatmap_animation{video_number}.mp4",
        sigma=15,
        color_map=cv2.COLORMAP_HOT,
        mask_path=f"output/masks/ball_heatmap_mask{video_number}.npy"
    )

    print(f"Ball Heatmap animation saved to: {ball_heatmap_path}")

    # Create score heatmap (combination of player and ball heatmaps)
    score_heatmap_path = mini_court.create_scoring_heatmap_animation(
    player_mini_court_detections, 
    ball_mini_court_detections, 
    ball_landing_frames,
    output_path=f"output/animations/scoring_heatmap_animation{video_number}.mp4", 
    color_map=cv2.COLORMAP_HOT
    )

    print(f"Score Heatmap animation saved to: {score_heatmap_path}")




    output_frames_real = write_track(video_frames_real, ball_detections_tracknet)
    output_frames_real = draw_ball_hits(output_frames_real, ball_shots_frames_audio) # Or ball_shots_frames


    # Draw keypoints, according to the first frame
    output_frames_real = courtline_detector.draw_keypoints_on_video(output_frames_real, refined_keypoints)



    # Draw Mini Court
    if DRAW_MINI_COURT:
        output_frames_real = mini_court.draw_mini_court(output_frames_real)
        output_frames_real = mini_court.draw_ball_landing_heatmap(
            output_frames_real,
            player_mini_court_detections,
            ball_mini_court_detections,
            ball_landing_frames,
            first_player_balls_frames,
            sigma=15
        )
        #output_frames_real = mini_court.draw_player_distance_heatmap(output_frames_real, player_mini_court_detections)
        #output_frames_real = mini_court.draw_ball_landing_points(output_frames_real, ball_mini_court_detections, ball_landing_frames)
        output_frames_real = mini_court.draw_shot_trajectories(
            output_frames_real, 
            player_mini_court_detections, 
            ball_mini_court_detections, 
            ball_landing_frames,
            first_player_balls_frames
        )     
                  
        output_frames_real = mini_court.draw_points_on_mini_court(output_frames_real, player_mini_court_detections, color = (255,255,0))
      #  output_frames_real = mini_court.draw_points_on_mini_court(output_frames_real, ball_mini_court_detections, color = (0,255,255))

    # Draw player stats
    if SAVE_STATS_BOX_SEPARATELY:
        # Create a separate video containing only the player stats box
        stats_box_path = create_player_stats_box_video(player_stats_data_df, video_number) # Here we are using the custom function to save the stats box separately
        print(f"Player Stats Box saved to: {stats_box_path}")
    else:
        # Draw player stats on the output video
        output_frames_real = draw_player_stats(output_frames_real, player_stats_data_df) # Here we are using the original function

    # Draw frame number (top left corner)
    for i, frame in enumerate(output_frames_real):
        cv2.putText(frame, f"Frame n {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save video
    save_video(output_frames_real, output_video_path, fps)
    

 




if __name__ == '__main__':
    main()


    '''
    

    # Draw frame number (top left corner) 
    for i, frame in enumerate(video_frames_real):
        cv2.putText(frame, f"Frame n {i}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    '''