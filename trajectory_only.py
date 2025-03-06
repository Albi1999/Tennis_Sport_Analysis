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
                   get_ball_shot_frames_visual,
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   create_black_video,
                   scraping_data,
                   splitting_data)
from trackers import (PlayerTracker, BallTracker, BallTrackerNetTRACE)
from mini_court import MiniCourt
from court_line_detector import CourtLineDetector
from ball_landing import BallLandingDetector
import cv2 
import torch 
from copy import deepcopy
import pandas as pd
import info




def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_number = 101
    input_video_path = f'data/input_video{video_number}.mp4'  # Toy example
    #input_video_path = f'data/videos/video_{video_number}.mp4' # Real example

    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
    convert_mp4_to_mp3(input_video_path, input_video_path_audio)
        
    output_video_path = f'output/trajectory_model_videos/output_video{video_number}.mp4'

    ball_tracker_method = 'tracknet' # 'yolo' or 'tracknet'
    
    if ball_tracker_method == 'yolo':
        ball_tracker = BallTracker(model_path = 'models/yolo11best.pt') # TODO : try yolo11last aswell as finetuning on more self annotated data (retrain model)
    
    if ball_tracker_method == 'tracknet':
        ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
        saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location= device)
        ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
        ball_tracker_TRACKNET.to(device)
        ball_tracker_TRACKNET.eval() 

    
    # Initialize Tracker for Players & Ball
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

    courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth') # TODO : add the postprocessing from this github : https://github.com/yastrebksv/TennisCourtDetector

    # Read in video
    video_frames, fps, video_width, video_height = read_video(input_video_path)

    # Detect & Track Players
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub = True,
                                                     stub_path = 'tracker_stubs/player_detections.pkl')


    if ball_tracker_method == 'tracknet':

        # TODO : fix the few points that are weirdly tracked ; 
        # should already be fixed with remove_outliers, but
        # maybe the interpolation then gets them back?
        # it seems to be easy fixable with just distance between tracks,
        # given we find some consistent track over some frames in the beginning

        # TODO : Put all of this in a class call like above
        # Calculate the correct scale factor for scaling back 
        # with TrackNet, we scaled to 640 width, 360 height

        import pickle
        stub_path = 'tracker_stubs/tracknet_ball_detections.pkl'
        dists_path =  'tracker_stubs/tracknet_ball_dists.pkl'
        read = True # read from stub
        if stub_path is not None and read == True:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            with open(dists_path, 'rb') as f:
                dists = pickle.load(f)
            
        if stub_path is not None and read == False: 
            scaling_x = video_width/640
            scaling_y = video_height/360
            ball_detections, dists = infer_model(video_frames, ball_tracker_TRACKNET, scale = (scaling_x, scaling_y))
            ball_detections = remove_outliers(ball_detections, dists)
            subtracks = split_track(ball_detections)
            for r in subtracks:
                ball_subtrack = ball_detections[r[0]:r[1]]
                ball_subtrack = interpolation(ball_subtrack)
                ball_detections[r[0]:r[1]] = ball_subtrack
        
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
            with open(dists_path, 'wb') as f:
                pickle.dump(dists, f)




        

        
    

            
        
        # Copy TrackNet ball_detections
        ball_detections_tracknet = ball_detections.copy()
        ball_detections = convert_ball_detection_to_bbox(ball_detections)


    # Detect court lines (on just the first frame, then they are fixed) # TODO : call this again whenever camera moves ? 
    refined_keypoints = courtline_detector.predict(video_frames[0])


    # Filter players
    player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(refined_keypoints, player_detections)
            
    mini_court = MiniCourt(video_frames[0])
    # Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                            ball_detections,
                                                                                                                            refined_keypoints,
                                                                                                                            chosen_players_ids)
    

    # Get racket hits 
    ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = False)
    ball_shots_frames = get_ball_shot_frames_visual(ball_mini_court_detections)
    ball_shots_frames = combine_visual_audio(ball_shots_frames, ball_shots_frames_audio, fps)

    print(ball_shots_frames)

    
    # Draw Output


    # First, create the completely black video
    frame_count = len(video_frames)
    input_video_black_path = f"data/trajectory_model_videos/output_video{video_number}.mp4"
    create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)
    

    # Read in this video
    video_frames, fps, video_width, video_height = read_video(input_video_black_path)


    # Draw ball tracking
    # Use both methods : circle/line
    if ball_tracker_method == 'tracknet':
        trace = 10
        # CHANGE HERE CIRCLE/LINE
        output_frames = write_track(video_frames, ball_detections_tracknet, ball_shots_frames, trace = trace, draw_mode= 'circle')

    
    # Draw Keypoints of court 
    output_frames = courtline_detector.draw_keypoints_on_video(output_frames, refined_keypoints)


    # Draw frame number (top left corner)
    #for i, frame in enumerate(output_frames):
    #    cv2.putText(frame, f"Frame n {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Scrape data
    # Please make a list for EACH video, in case we have to rerun for some reason (that we then dont have to )
    ball_ground_hits_v_101 = [45,75,102,131,163,195,222,253,295]


    output_path_circle = 'data/trajectory_model_dataset/circles'
    output_path_line = 'data/trajectory_model_dataset/lines'

    # CHANGE HERE PATH
 #   scraping_data(video_n = 101, output_path= output_path_circle, input_frames= output_frames, ball_bounce_frames= ball_ground_hits_v_101, ball_shots_frames = ball_shots_frames, trace = trace)

    # change accordingly if on line or on circles
  #  train,val,test = splitting_data(main_dir = 'data/trajectory_model_dataset/circles')

    # Save video
    save_video(output_frames, output_video_path, fps)



if __name__ == '__main__':
    main()
