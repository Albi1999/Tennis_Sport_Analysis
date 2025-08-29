from utils import (read_video, 
                   refine_audio, 
                   write_track, 
                   get_ball_shot_frames_audio, 
                   convert_mp4_to_mp3,
                   create_black_video,
                   scraping_data_for_inference,
                   detect_frames_TRACKNET,
                   get_ball_shot_frames_visual,
                   combine_audio_visual,
                   cluster_series)
from trackers import (PlayerTracker, BallTrackerNetTRACE)
from ball_landing import (BounceCNN, make_prediction, evaluation_transform)
from court_line_detector import CourtLineDetector
import torch 
import os 
import pickle
import numpy as np



def main():

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Video to run inference on (Change video_number and ground_truth_bounce)
    video_number = 116
    ground_truth_bounce = [12, 51, 81]
    print(f"Running inference on video {video_number}")

    input_video_path = f'data/input_video{video_number}.mp4' 
    output_video_path = f'output/trajectory_model_videos/output_video{video_number}.mp4'

    # Convert video to mp3
    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
    convert_mp4_to_mp3(input_video_path, input_video_path_audio)
        
    # Check if we already processed that video by looking if output with video number reference exists (for faster testing)
    if os.path.exists(output_video_path):
        READ_STUBS = True
    else:
        READ_STUBS = False

    # Initialize Ball Tracker (TrackNet)
    ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
    saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location= device, weights_only=False)
    ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
    ball_tracker_TRACKNET.to(device)
    ball_tracker_TRACKNET.eval() 

    # Initialize Player Tracker
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

    # Initialize Courtline Detector
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
            
    # Get Racket Hits based on audio


    ball_shots_frames_visual = get_ball_shot_frames_visual(ball_detections_tracknet, fps)
    ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)

    ball_shots_frames = combine_audio_visual(ball_shots_frames_visual= ball_shots_frames_visual,
                                                ball_shots_frames_audio= ball_shots_frames_audio, 
                                                fps = fps, 
                                                max_distance_param = 7)
    


    print(f"Ball Shot Frames : {ball_shots_frames}")

    ## Draw Output ##


    # First, create a completely black video with same dimensions & fps of actual video 
    frame_count = len(video_frames_real)
    input_video_black_path = f"data/trajectory_model_videos/output_video{video_number}.mp4"  # on black video (needed for bounce detection)
    create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)

    # Read in this video
    video_frames_black, fps, video_width, video_height = read_video(input_video_black_path)

    # Draw Ball Detection into black video
    output_frames_black = write_track(video_frames_black, ball_detections_tracknet, ball_shots_frames, trace = 10, draw_mode= 'circle')

    # Draw Keypoints (and lines) of the court into black video 
    output_frames_black = courtline_detector.draw_keypoints_on_video(output_frames_black, refined_keypoints)

    _ = scraping_data_for_inference(video_n= video_number, output_path = 'data_inference', input_frames = output_frames_black,
                                 ball_shots_frames = ball_shots_frames , trace = 10, ball_detections = ball_detections_tracknet)

    # Instantiate Bounce Model
    model_bounce = BounceCNN()
    with open('data_bounce_stubs/data_mean_std.pkl', 'rb') as f:
        data_mean_and_std = pickle.load(f)
    
    mean = data_mean_and_std[0]
    std = data_mean_and_std[1]

    # Make Predictions
    predictions, confidences, img_idxs = make_prediction(model = model_bounce, best_model_path = 'models/best_bounce_model.pth',
                                         input_frames_directory = 'data_inference/images_inference', transform = evaluation_transform(mean, std), device = device)
    
    mask = np.array(predictions) == 1
    img_idxs_bounce = np.array(img_idxs)[mask].tolist()
    ball_landing_frames = cluster_series(img_idxs_bounce)

    print(f"Predicted V Shaped Frames : {img_idxs_bounce}")
    print(f"Predicted Bounce Frames : {ball_landing_frames}")
    print(f"Ground Truth Bounce Frames : {ground_truth_bounce[1:]}")
    


if __name__ == '__main__':
    main()

