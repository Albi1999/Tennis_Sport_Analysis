from utils import (read_video, 
                   save_video, 
                   infer_model, 
                   remove_outliers, 
                   split_track, 
                   combine_visual_audio, 
                   interpolation, 
                   write_track, 
                   get_ball_shot_frames_audio, 
                   convert_mp4_to_mp3,
                   draw_ball_hits, 
                   convert_ball_detection_to_bbox, 
                   get_ball_shot_frames_new,
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats)
from trackers import (PlayerTracker, BallTracker, BallTrackerNetTRACE)
from mini_court import MiniCourt
from court_line_detector import CourtLineDetector
import cv2 
import torch 
from copy import deepcopy
import pandas as pd
import info


def main():

    AUDIO = True
    DRAW_MINI_COURT = False
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_number = 101
    input_video_path = f'data/input_video{video_number}.mp4'  # Toy example
    #input_video_path = f'data/videos/video_{video_number}.mp4' # Real example
    
    if AUDIO:
        input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
        convert_mp4_to_mp3(input_video_path, input_video_path_audio)
        
    output_video_path = f'output/output_video{video_number}.mp4'

    # Initialize Tracker for Players & Ball
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')
    ball_tracker_method = 'yolo' # 'yolo' or 'tracknet'
    
    if ball_tracker_method == 'yolo':
        ball_tracker = BallTracker(model_path = 'models/yolo11best.pt') # TODO : try yolo11last aswell as finetuning on more self annotated data (retrain model)
    
    elif ball_tracker_method == 'tracknet':
        ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
        saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location= device)
        ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
        ball_tracker_TRACKNET.to(device)
        ball_tracker_TRACKNET.eval() 
    
    else:
        raise ValueError("Specify a valid ball tracker method ('yolo' or 'tracknet')")
    
    courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth') # TODO : add the postprocessing from this github : https://github.com/yastrebksv/TennisCourtDetector
    

    # Read in video
    video_frames, fps, video_width, video_height = read_video(input_video_path)

    # Detect & Track Players
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub = True,
                                                     stub_path = 'tracker_stubs/player_detections.pkl')

    
    # Detect Ball 
    if ball_tracker_method == 'yolo':
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                    read_from_stub = True,
                                                    stub_path = 'tracker_stubs/ball_detections.pkl')
        # Interpolate the missing tracking positions for the ball
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)


        if AUDIO:
            ball_shots_frames_audio = ball_tracker.get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)





    elif ball_tracker_method == 'tracknet':
        # Calculate the correct scale factor for scaling back 
        # with TrackNet, we scaled to 640 width, 360 height
        scaling_x = video_width/640
        scaling_y = video_height/360
        ball_detections, dists = infer_model(video_frames, ball_tracker_TRACKNET, scale = (scaling_x, scaling_y))
        ball_detections = remove_outliers(ball_detections, dists)
        subtracks = split_track(ball_detections)
        for r in subtracks:
            ball_subtrack = ball_detections[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_detections[r[0]:r[1]] = ball_subtrack
    
        if AUDIO:
            ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)
            
        
        # Copy TrackNet ball_detections
        ball_detections_tracknet = ball_detections.copy()
        ball_detections = convert_ball_detection_to_bbox(ball_detections)
        print(f"Ball detections: {ball_detections[40]}")    # Example of how to access ball detections at frame 40 (gives (x,y) coordinate)
            
    



    # Detect court lines (on just the first frame, then they are fixed) # TODO : call this again whenever camera moves ? 
    courtline_keypoints = courtline_detector.predict(video_frames[0])


    # Filter players
    player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(courtline_keypoints, player_detections)

    
    #print(f"Player detections: {player_detections[40][1]}")    # Example of how to access player detections at frame 40




    # Initialize MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shots_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(f"Ball shots detected at frames: {ball_shots_frames}")
    # first_player_balls_frames = ball_shots_frames[0::2]
    
    first_player_balls_frames = [52, 113, 177, 235, 305] # Hardcoded for now
    print(f"First player ball shots detected at frames: {first_player_balls_frames}")
    
    # List of frames where ball hits the ground
    ball_landing_frames = [45, 75, 102, 130, 165, 194, 223, 255, 296, 324] # Hardcoded for now
    print(f"Ball landing frames: {ball_landing_frames}")

    
    # Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                            ball_detections,
                                                                                                                            courtline_keypoints,
                                                                                                                            chosen_players_ids)
    
    # Speed stats

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

    # Check the racket hits

    #ball_shots_frames = get_ball_shot_frames_new(ball_mini_court_detections)

    #ball_shots_frames = combine_visual_audio(ball_shots_frames, ball_shots_frames_audio)


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

    
    # Draw Output

    # Draw bounding boxes around players + their IDs
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)



    # Draw bounding box around ball
    if ball_tracker_method == 'yolo':
        output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
        output_frames = ball_tracker.draw_ball_hits(output_frames, ball_shots_frames_audio)  # Or ball_shots_frames 
    elif ball_tracker_method == 'tracknet':
        output_frames = write_track(output_frames, ball_detections_tracknet)
        output_frames = draw_ball_hits(output_frames, ball_shots_frames_audio) # Or ball_shots_frames
    
    # Draw keypoints, according to the first frame
    output_frames = courtline_detector.draw_keypoints_on_video(output_frames, courtline_keypoints)
    
 

    # Draw Mini Court
    if DRAW_MINI_COURT:
        output_frames = mini_court.draw_mini_court(output_frames)
        output_frames = mini_court.draw_ball_landing_heatmap(
            output_frames,
            player_mini_court_detections,
            ball_mini_court_detections,
            ball_landing_frames,
            first_player_balls_frames,
            sigma=15
        )
        #output_frames = mini_court.draw_player_distance_heatmap(output_frames, player_mini_court_detections)
        #output_frames = mini_court.draw_ball_landing_points(output_frames, ball_mini_court_detections, ball_landing_frames)
        output_frames = mini_court.draw_shot_trajectories(
            output_frames, 
            player_mini_court_detections, 
            ball_mini_court_detections, 
            ball_landing_frames,
            first_player_balls_frames
        )     
                  
        output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections, color = (255,255,0))
      #  output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, color = (0,255,255))

    # Draw player stats
    output_frames = draw_player_stats(output_frames, player_stats_data_df)

    # Draw frame number (top left corner)
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame n {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save video
    save_video(output_frames, output_video_path, fps)



if __name__ == '__main__':
    main()
