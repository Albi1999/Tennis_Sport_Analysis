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
                   splitting_data,
                   detect_frames_TRACKNET)
from trackers import (PlayerTracker, BallTracker, BallTrackerNetTRACE)
from mini_court import MiniCourt
from court_line_detector import CourtLineDetector
import cv2 
import torch 
from copy import deepcopy
import pandas as pd
import info
import os 





def main():

    video_numbers = [100]

    for video_number in video_numbers:
   # video_number = 100


        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_video_path = f'data/input_video{video_number}.mp4'  # Toy example
        #input_video_path = f'data/videos/video_{video_number}.mp4' # Real example

        input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
        convert_mp4_to_mp3(input_video_path, input_video_path_audio)
            
        output_video_path = f'output/trajectory_model_videos/output_video{video_number}.mp4'


        # Check if we already processed that video by looking if output with video number reference exists
        if os.path.exists(output_video_path):
            READ_STUBS = True
        else:
            READ_STUBS = False

        ball_tracker_method = 'tracknet' # 'yolo' or 'tracknet'
        
        if ball_tracker_method == 'yolo':
            ball_tracker = BallTracker(model_path = 'models/yolo11best.pt')
        
        if ball_tracker_method == 'tracknet':
            ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
            saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location= device)
            ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
            ball_tracker_TRACKNET.to(device)
            ball_tracker_TRACKNET.eval() 

        
        # Initialize Tracker for Players & Ball
        player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

        courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth') 

        # Read in video
        video_frames_real, fps, video_width, video_height = read_video(input_video_path)

        # Detect & Track Players
        player_detections = player_tracker.detect_frames(video_frames_real,
                                                        read_from_stub = READ_STUBS,
                                                        stub_path = f'tracker_stubs/player_detections_{video_number}.pkl')


        if ball_tracker_method == 'tracknet':

            ball_detections, ball_detections_tracknet = detect_frames_TRACKNET(video_frames_real, video_number = video_number, tracker =ball_tracker_TRACKNET,
                                video_width=video_width, video_height= video_height, read_from_stub = READ_STUBS, 
                                stub_path=  f'tracker_stubs/tracknet_ball_detections_{video_number}.pkl')

        



        # Detect court lines (on just the first frame, then they are fixed) 
        refined_keypoints = courtline_detector.predict(video_frames_real[0])


        # Filter players
        player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(refined_keypoints, player_detections)
                
        mini_court = MiniCourt(video_frames_real[0])
        # Convert player positions to mini court positions
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                                ball_detections,
                                                                                                                                refined_keypoints,
                                                                                                                                chosen_players_ids)
        

        # Get racket hits 
        ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = True)
        ball_shots_frames = refine_audio(ball_shots_frames_audio, fps, input_video_path_audio)


        print("Ball Shots from Audio : ", ball_shots_frames_audio)
        print("Audio Refinment :", ball_shots_frames)

        
        # Draw Output

        # First, create the completely black video
        frame_count = len(video_frames_real)
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
        for i, frame in enumerate(video_frames_real):
            cv2.putText(frame, f"Frame n {i}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)


        output_path_circle = 'data/trajectory_model_dataset/circles'
        #output_path_line = 'data/trajectory_model_dataset/lines'


        # video 105 : racket hit at 186 (missed by model)
        # video 107 : remove the last (269, because of voice of commentator seen as a peak), and also remove 203 
        # video 108 : Shot at 42 is the racket hitting the ground, the real shot was at 37 - last shot was at 91 instead of 89
        # video 109 : at 47 there is no shot, shot at 61 is at 65, 116 is at 124, 148 is at 153, 171 is at 177, missed shot at 211 

        # Look into output/trajectory_model_videos and look at the _frames videos : here we can see the frame the ball hits the ground
        # If the ball is too occluded : look into the output/trajectory_model_videos folder and there the normal output videos :
        # see if the trajectory ("V" shape) is still visible ; else add "None"
        if video_number == 100:
            ball_bounce_frames = [49, 75, 106, 142]
        if video_number == 101:
            # removed 169 because trace is just a single point ; bad for training
            ball_bounce_frames = [20,50,76,106,138,197,230,273,301]
        if video_number == 102:
            ball_bounce_frames = [13,41,73,104,131,159,190,221,270,299,329,363,414]
        if video_number == 103:
            ball_bounce_frames = [23,66,97]
        if video_number == 105:
            ball_bounce_frames = [48,None,115,146,208,268]
            ball_shots_frames.append(186)

        if video_number == 107:
            ball_bounce_frames = [69,107,133,172,202,279]
            ball_shots_frames.remove(269)
            ball_shots_frames.remove(203)

        if video_number == 108:
            ball_bounce_frames = [29, 78, 104]
            ball_shots_frames.append(37)
            ball_shots_frames.remove(42)

        if video_number == 109:
            ball_bounce_frames = [30, 55, 86, 111, 144, 169, 203, 244]
            ball_shots_frames.remove(47)
            ball_shots_frames.remove(61)
            ball_shots_frames.append(65)
            ball_shots_frames.remove(116)
            ball_shots_frames.append(124)
            ball_shots_frames.remove(148)
            ball_shots_frames.append(153)
            ball_shots_frames.remove(171)
            ball_shots_frames.append(177)
            ball_shots_frames.append(211)


        ball_shots_frames = sorted(ball_shots_frames)


        

        # CHANGE HERE PATH
        scraping_data(video_n = video_number, output_path= output_path_circle, input_frames= output_frames, ball_bounce_frames= ball_bounce_frames, ball_shots_frames = ball_shots_frames, trace = trace, ball_detections = ball_detections_tracknet)

        # change accordingly if on line or on circles
   #     _,_,_ = splitting_data(main_dir = 'data/trajectory_model_dataset/circles')
    


        # Save video
    #    save_video(output_frames, output_video_path, fps)
    #    save_video(video_frames_real, f'output/trajectory_model_videos/output_video{video_number}_frames.mp4', fps)






if __name__ == '__main__':
    main()
