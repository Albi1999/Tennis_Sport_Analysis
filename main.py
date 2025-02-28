from utils import (read_video, save_video, postprocess, infer_model, remove_outliers, split_track, interpolation, write_track, convert_mp4_to_mp3)
from trackers import (PlayerTracker, BallTracker,BallTrackerNet)
from mini_court import MiniCourt
from court_line_detector import CourtLineDetector
import cv2 
import torch 



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
        ball_tracker_TRACKNET = BallTrackerNet()
        ball_tracker_TRACKNET.load_state_dict(torch.load('models/tracknet_best.pt', map_location= device))
        ball_tracker_TRACKNET.to(device)
        ball_tracker_TRACKNET.eval()
        
    # TODO: ADD THE TRACKNET MODEL + YOLO MODEL    
    # elif ball_tracker_method == 'both':
        
    else:
        raise ValueError("Specify a valid ball tracker method ('yolo' or 'tracknet')")
    
    courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth') # TODO : add the postprocessing from this github : https://github.com/yastrebksv/TennisCourtDetector
    

    # Read in video
    video_frames, fps = read_video(input_video_path)

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
            ball_shots_frames = ball_tracker.get_ball_shot_frames_audio(input_video_path_audio, fps, plot = False)





    elif ball_tracker_method == 'tracknet':
        ball_detections, dists = infer_model(video_frames, ball_tracker_TRACKNET)
        ball_detections = remove_outliers(ball_detections, dists)
        subtracks = split_track(ball_detections)
        for r in subtracks:
            ball_subtrack = ball_detections[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_detections[r[0]:r[1]] = ball_subtrack



    # Detect court lines (on just the first frame, then they are fixed) # TODO : call this again whenever camera moves ? 
    courtline_keypoints = courtline_detector.predict(video_frames[0])


    # Filter players
    player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(courtline_keypoints, player_detections)

    
    #print(f"Player detections: {player_detections[40][1]}")    # Example of how to access player detections at frame 40

    # Initialize MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
   # ball_shots_frames = ball_tracker.get_ball_shot_frames(ball_detections)
   # print(f"Ball shots detected at frames: {ball_shots_frames}")
    
    # Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                            ball_detections,
                                                                                                                            courtline_keypoints,
                                                                                                                            chosen_players_ids)
    
    # Create Heatmap Animation
    heatmap_animation_path = mini_court.create_heatmap_animation(player_mini_court_detections,
                                                                output_path=f"output/animations/heatmap_animation{video_number}.mp4")
    print(f"Heatmap animation saved to: {heatmap_animation_path}")

    # Draw Output

    # Draw bounding boxes around players + their IDs
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)



    # Draw bounding box around ball
    if ball_tracker_method == 'yolo':
        output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
        output_frames = ball_tracker.draw_ball_hits(output_frames, ball_shots_frames)
    elif ball_tracker_method == 'tracknet':
        output_frames = write_track(output_frames, ball_detections)
    
    # Draw keypoints, according to the first frame
    output_frames = courtline_detector.draw_keypoints_on_video(output_frames, courtline_keypoints)

    # Draw Mini Court
    if DRAW_MINI_COURT:
        output_frames = mini_court.draw_mini_court(output_frames)
        output_frames = mini_court.draw_player_distance_heatmap(output_frames, player_mini_court_detections)
        output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections, color = (255,255,0))
        output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, color = (0,255,255))

    
    # Draw frame number (top left corner)
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame n {i}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save video
    save_video(output_frames, output_video_path, fps)


if __name__ == '__main__':
    main()
