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
                   convert_ball_detection_to_bbox, 
                   get_ball_shot_frames_visual,
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   remove_outliers_final,
                   combine_audio_visual,
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
    SCRAPING = False
    # TAKE :

    #
    # 100
    # yolo ball tracking
    # Ground Truth : [40, 61, 85, 116]
    # Ours : [40,61,85,119]
    # 119 instead of 116 (neglectable mistake) (0 mistakes)

    # 101
    # yolo ball tracking
    # Ground Truth : [12,28,58,89,115,153,179,210,243,284]
    # Ours : [12,28, 58, 89, 115, 123, 153, 179, 211, 243, 283]
    # additional 123 (1 mistake)


    # 102 (perfect blue court video)
    # yolo ball tracking 
    # Ground Truth : [4, 21, 52, 82, 109, 141, 169, 201, 236, 282, 310, 343, 372, 427]
    # Ours : [4, 21, 52, 82, 109, 141, 169, 201, 236, 282, 310, 343, 372, 427]
    # (0 mistakes)

    # 103 
    # yolo ball tracking
    # Ground Truth : [17, 34, 85]
    # Ours : [17, 85]
    # misses 34 (since ball tracking fails there a bit) (1 mistake)

    # 105
    # yolo ball tracking
    # Ground Truth : [40, 56,91,127,164,189,224,259]
    # Ours : [33, 64, 91, 127, 164, 259]
    # 189 not caught due to too little audio (but visual caught it), 224 not caught because taken in cluster with 259 (ball tracking
    # there too inconsistent and therefore shows both balls on same court side, even though not true)
    # (4 mistakes)

    # 107
    # fails since player tracking fails at one point 



    # 108
    # yolo ball tracking
    # Ground Truth : [21, 42, 89]
    # Ours : [21, 42, 89]
    # (0 mistakes)


    # 109
    # yolo ball tracking
    # Ground Truth : [21, 35, 61, 90, 125, 154, 176, 211]
    # Ours : [21, 35, 61, 90, 116, 154, 171, 204, 249]
    # mistakes due to shoe sliding (picked up by audio) : possible fix : check where audio signal is stronger
    # for a longer duration (so not just a single peak), can be indicator for shoe sliding
    # picked up 249 due to ball tracking failure
    # (4 mistakes)


    # 110
    # yolo ball tracking
    # Ground Truth : [14,33,68, 103,136,176,211,254...]
    # Ours : [5, 33, 68, 101, 209, 236, 277, 299, 339, 375, 406, 414]
    # way too many mistakes



    # 111 (perfect green court video)
    # yolo ball tracking
    # Ground Truth : [36,52,86,120,153,186,214,247]
    # Ours : [36, 52, 86, 120, 153, 186, 214, 247]
    # (0 mistakes)


    # 116 (perfect clay court video, we just need to cut out the last part where it detects 95 wrongly)
    # tracknet ball tracking for mini court coordinates, but yolo tracking for the visual direction change model
    # Ground Truth : [1, 23, 65, 95]
    # Ours : [1, 23, 65]
    # (1 mistake that we can cherry pick into 0)





    counter  = 0
    # Change here which videos to get data from
    video_numbers = [i for i in range(1000,1015)] #[100,101,102,103,105,107,108,109,110,111,112,113,114,115,116,117,118]

    for video_number in video_numbers:



        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_video_path = f'data/new_input_videos/input_video_{video_number}.mp4'  # Toy example
        #input_video_path = f'data/videos/video_{video_number}.mp4' # Real example

        input_video_path_audio = f'data/new_input_videos/input_video_{video_number}_audio.mp3'
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


          #  ball_detections_YOLO = ball_tracker_yolo.detect_frames(video_frames_real, 
          #                                                         read_from_stub = True, 
          #                                                         stub_path = f'tracker_stubs/ball_detections_YOLO_{video_number}.pkl')
            
          #  ball_detections_YOLO = ball_tracker_yolo.interpolate_ball_positions(ball_detections_YOLO)
        
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

            ball_tracker_yolo = BallTracker(model_path = 'models/yolo11best.pt')


            ball_detections_YOLO = ball_tracker_yolo.detect_frames(video_frames_real, 
                                                                   read_from_stub = READ_STUBS, 
                                                                   stub_path = f'tracker_stubs/ball_detections_YOLO_{video_number}.pkl')
            
            ball_detections_YOLO = ball_tracker_yolo.interpolate_ball_positions(ball_detections_YOLO)


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
        mini_court_keypoints = mini_court.drawing_key_points


        # Get the first hit with less refined audio finding

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


        # Draw Output

        # First, create the completely black video
        frame_count = len(video_frames_real)
        input_video_black_path = f"data/trajectory_model_videos/output_video{video_number}.mp4"
        create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)
        

        # Read in this video
        video_frames, fps, video_width, video_height = read_video(input_video_black_path)


        # Draw ball tracking
        if ball_tracker_method == 'tracknet':
            trace = 5 # CHANGE TRACE HERE
            output_frames = write_track(video_frames, ball_detections_tracknet, ball_shots_frames, trace = trace, draw_mode= 'circle')

        
        # Draw Keypoints of court 
      #  output_frames = courtline_detector.draw_keypoints_on_video(output_frames, refined_keypoints)


        # Draw frame number (top left corner) 
        for i, frame in enumerate(video_frames_real):
            cv2.putText(frame, f"Frame n {i}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)


        output_path_circle = 'data/trajectory_model_dataset/circles'
        #output_path_line = 'data/trajectory_model_dataset/lines'


        # video 105 : racket hit at 186 (missed by model)
        # video 107 : remove the last (269, because of voice of commentator seen as a peak), and also remove 203 
        # video 108 : Shot at 42 is the racket hitting the ground, the real shot was at 37 - last shot was at 91 instead of 89
        # video 109 : at 47 there is no shot, shot at 61 is at 65, 116 is at 124, 148 is at 153, 171 is at 177, missed shot at 211 
        # video 110 : from frame 90 it becomes a mess
        # video 111 : no shot at 274
        # video 112 : no shot at 100, 167, 317, 466 but shot at 158, 193, 445
        # video 113 : no shot at 144, missed shot at 33, 76
        # video 114 : no shot at 65, 153, 219, 249
        # video 115 : no shot at 44, 169, missed shot at 156

        # Look into output/trajectory_model_videos and look at the _frames videos : here we can see the frame the ball hits the ground
        # If the ball is too occluded : look into the output/trajectory_model_videos folder and there the normal output videos :
        # see if the trajectory ("V" shape) is still visible ; else add "None"

        # b
        ball_bounce_frames_hardc = [[76],[56,92],[23,51,87], [20,49,76,101,141],[36,67,101],
                                     [18,56,86], [14,48,80], [11,51,80,119,150,180,211,238,271,312], [16,48,79,109,139,172,201],[3, 37,68],
                                     [23, 54, 84, 127, 152, 183, 213, 243, 267, 304, 335, 384, 417, 444, 478, 504,530, 565, 592, 636, 664, 710,739, 768,805,835],
                                     [19, 58], [34, 65, 91], [34, 63, 97, 129], [26, 65, 94], [28, 56, 83, 121, 149, 182, 214, 241, 268, 296, 335, 364],
                                     [17, 49], [30, 60], [11, 46, 77], [26, 61], [149, 177, 206], [32, 65, 91, 130], [33, 59, 90, 124, 156, 186, 222, 252, 285, 313],
                                     [17, 57, 85, 117, 142, 185, 213, 245, 271, 303], [19, 51, 95, 128], [30, 66, 108], [36, 69, 96, 133, 164, 196, 229],
                                     [32,75], [29,56, 86, 111,174,220], [36,71,104,131,167,195], [31,67,99,137,170,199,235,270,298,324,357,384,428,472,521,552],
                                     [29,74,99,129], [29,65], [33,65,95,131,163,191], [32,67,96,143,170,203,230,266,291,324,352,399,428], [35, 65, 79, 106],
                                     [28, 61, 92, 127, 152, 182, 214, 240, 277, 308, 338, 363, 400], [34, 62], [42, 67, 96], [28, 57, 87]
                                     ]
        
        # r
        ball_shots_frames_hardc = [[64,84],[43,65],[13,30,60], [9,27,52,84,115], [24,45,78],[8,27,71], 
                                   [5,22,58], [3,20,62,94,126,157,189,219,252,285], [8,26,61,90,119,148,179,212], [12, 47],
                                   [10,33, 65,98,136,166, 191, 222, 237, 286, 317, 351, 392, 424, 456, 485, 516, 543, 572, 605, 647, 677, 718, 750, 784, 817],
                                   [10, 28, 72], [24, 44, 74],[22, 46, 73, 108], [18, 35, 80], [18, 35, 67, 95, 130, 163, 194, 220, 252, 276, 306, 346],
                                   [7, 26], [20, 37, 72], [1, 21, 54, 81], [17, 33], [138, 157, 189], [23, 40, 75, 102], [20, 42, 69, 103, 132, 166, 190, 233, 262, 292, 321],
                                   [8, 26, 69, 97, 123, 155, 193, 223, 253, 281], [10, 28, 59, 111], [20, 39, 81], [23, 46, 74, 108, 137, 177, 207, 238],
                                   [23,40,86],[17, 37,60, 97, 125, 147, 195],[24, 49, 82,111,147,174],[20,36,80,112,144,178,213,246,276,305,337,365,395,439,492,534],
                                   [20,40,85,104],[18,35],[24,41,72,103,137,173,204],[23,40,78,105,152,178,213,241,275,304,336,363,409], [23, 56, 86, 118],
                                   [20, 37, 73, 98, 134, 164, 194, 220, 256, 286, 317, 344, 377], [22, 43, 71], [32, 49, 78], [18, 37, 64]
                                   ]

        # 1030 : r [23,40,86] b [32,75,]
        # 1032 : r [17, 37,60, 97, 125, 147, 195] b [29,56, 86, 111,174,220]
        # 1033 : r [24, 49, 82,111,147,174] b [36,71,104,131,167,195]
        # 1034 : r [20,36,80,112,144,178,213,246,276,305,337,365,395,439,492,534] b [31,67,99,137,170,199,235,270,298,324,357,384,428,472,521,552]
        # 1035 : r [20,40,85,104] b [29,74,99,129]
        # 1036 : r [18,35] b [29,65]
        # 1037 : r [24,41,72,103,137,173,204] b [33,65,95,131,163,191,]
        # 1038 : r [23,40,78,105,152,178,213,241,275,304,336,363,409] b [32,67,96,143,170,203,230,266,291,324,352,399,428]
        
        # 1039 : r [23, 56, 86, 118] b [35, 65, 79, 106]
        # 1040 : r [20, 37, 73, 98, 134, 164, 194, 220, 256, 286, 317, 344, 377] b [28, 61, 92, 127, 152, 182, 214, 240, 277, 308, 338, 363, 400]
        # 1041 : missing
        # 1042 : r [22, 43, 71] b [34, 62]
        # 1043 : r [32, 49, 78] b [42, 67, 96]
        # 1044 : r [18, 37, 64] b [28, 57, 87]
        """
        if video_number == 100:
            ball_bounce_frames = [49, 75, 106, 142]
        if video_number == 101:
            # removed 169 because trace is just a single point ; bad for training
            ball_bounce_frames = [20,50,77,106,138,197,230,270,301] # [20,50,77,106,138,168,197,230,270,301]
        if video_number == 102:
            ball_bounce_frames = [13,41,73,105,131,159,190,221,270,299,329,363,414]
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

        if video_number == 110:
            ball_bounce_frames = [23, 60, 94, 123, 162, 200, 244, 273, 305, 334, 363, 390]
            ball_shots_frames.remove(90)
            ball_shots_frames.append(104)
            ball_shots_frames.remove(112)
            ball_shots_frames.remove(132)
            ball_shots_frames.append(136)
            ball_shots_frames.remove(169)
            ball_shots_frames.append(174)
            ball_shots_frames.remove(203)
            ball_shots_frames.append(213)
            ball_shots_frames.remove(229)
            ball_shots_frames.remove(243)
            ball_shots_frames.append(252)
            ball_shots_frames.remove(270)
            ball_shots_frames.append(282)
            ball_shots_frames.remove(299)
            ball_shots_frames.append(310)
            ball_shots_frames.remove(327)
            ball_shots_frames.append(340)
            ball_shots_frames.remove(359)
            ball_shots_frames.append(372)
            ball_shots_frames.remove(385)
            ball_shots_frames.append(400)
            ball_shots_frames.remove(414)
        
        if video_number == 111:
            ball_bounce_frames = [43, 76, 114, 144, 171, 207, 236]
            ball_shots_frames.remove(274)
        
        if video_number == 112:
            ball_bounce_frames = [45, 83, 109, 150, 178, 224, 268, 298, 330, 363, 405, 438]
            ball_shots_frames.remove(100)
            ball_shots_frames.append(158)
            ball_shots_frames.remove(167)
            ball_shots_frames.append(193)
            ball_shots_frames.remove(317)
            ball_shots_frames.append(445)
            ball_shots_frames.remove(466)
        
        if video_number == 113:
            ball_bounce_frames = [26, 67, 96, 133, 148]
            ball_shots_frames.append(33)
            ball_shots_frames.append(76)
            ball_shots_frames.remove(144)
        
        if video_number == 114:
            ball_bounce_frames = [19, 58, 83, 123, 152, 228]
            ball_shots_frames.remove(65)
            ball_shots_frames.append(69)
            ball_shots_frames.remove(153)
            ball_shots_frames.remove(219)
            ball_shots_frames.remove(249)

        if video_number == 115:
            ball_bounce_frames = [17, 50, 75, 120, 150, 183, 213, 242]
            ball_shots_frames.remove(44)
            ball_shots_frames.append(156)
            ball_shots_frames.remove(169)

        if video_number == 116:
            ball_bounce_frames = [12, 51, 81]
        
        if video_number == 117:
            ball_bounce_frames = [14, 45, 74, 104, 138, 165, 196]
            ball_shots_frames.remove(98)
        
        if video_number == 118:
            ball_bounce_frames = [38, 69, 95, 145, 171, 208, 237, 281, 309]
            ball_shots_frames.append(26)
            ball_shots_frames.append(45)
            ball_shots_frames.append(74)
            ball_shots_frames.remove(95)
            ball_shots_frames.append(109)
            ball_shots_frames.remove(225)
            ball_shots_frames.remove(321)
                # this one had a really bad prediction:
                # Output of the full_model_run.py:
                # ['175', '176', '177', '211', '212', '213', '238', '239', '240', '241', '242', '243', '244', '254',
                #  '255', '256', '257', '258', '259', '260', '282', '283', '284', '285', '286', '287', '292', '312', '313', '314', '315']

        ball_shots_frames = sorted(ball_shots_frames)
        """

        
    
        # CHANGE HERE PATH
        if SCRAPING:
            scraping_data(video_n = video_number, output_path= output_path_circle, input_frames= output_frames, ball_bounce_frames= ball_bounce_frames_hardc[counter], ball_shots_frames = ball_shots_frames_hardc[counter], trace = trace, ball_detections = ball_detections_tracknet)

        counter += 1
  
        if not SCRAPING:
            train_videos = [1000,1001,1002,1003,1004,1005,1006,1007,1008,1010]
            val_videos = [1009,1011]
            test_videos = [1012,1013,1014]
            _,_,_ = splitting_data(main_dir = 'data/trajectory_model_dataset/circles', train_videos = train_videos, val_videos = val_videos, test_videos = test_videos)
            break
    


        # Save video
    #    save_video(output_frames, output_video_path, fps)

        
    #    output_frames_real = mini_court.draw_mini_court(video_frames_real)
    #    output_frames_real = mini_court.draw_points_on_mini_court(output_frames_real, ball_mini_court_detections, color = (0,255,255))
    ##    save_video(output_frames_real, f'output/trajectory_model_videos/output_video{video_number}_frames.mp4', fps)






if __name__ == '__main__':
    main()
