
AUDIO = True

device = "cuda" if torch.cuda.is_available() else "cpu"

video_number = 101
input_video_path = f'data/input_video{video_number}.mp4'  # Toy example
if AUDIO:
    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
#input_video_path = f'data/videos/video_{video_number}.mp4' # Real example
output_video_path = f'output/output_video{video_number}.mp4'



# Detect Ball 
if ball_tracker_method == 'yolo':
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                read_from_stub = True,
                                                stub_path = 'tracker_stubs/ball_detections.pkl')
    # Interpolate the missing tracking positions for the ball
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    if AUDIO:
        ball_sound_detections = ball_tracker.get_ball_shot_frames_audio(input_video_path_audio)




from ultralytics import YOLO
import cv2
import pickle 
import pandas as pd 
import librosa 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfilt




    def get_ball_shot_frames_audio(self, audio_file, plot = True):
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
                            distance=int(0.3 * sr),  # Minimum distance between peaks
                            prominence=0.01)  # Find all distinct peaks 
        
        # Convert peak positions to time (seconds)
        hit_times = peaks / sr
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Plot filtered waveform with detected hits
            plt.subplot(2, 1, 1)
            times = np.linspace(0, len(y_filtered)/sr, len(y_filtered))
            plt.plot(times, y_filtered)
            plt.vlines(hit_times, -0.2, 0.2, color='r', linewidth=1)
            plt.title('Filtered Audio Waveform (150Hz-1800Hz) with Detected Hits')
            plt.xlabel('Time (s)')
            
            # Plot the envelope with detected peaks
            plt.subplot(2, 1, 2)
            plt.plot(times, y_envelope)
            plt.vlines(hit_times, 0, np.max(y_envelope), color='r', linewidth=1, label='Detected Hits')
            plt.title('Signal Envelope with Detected Peaks')
            plt.xlabel('Time (s)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("AUDIO.png")
           # plt.show()
        
        print(f"Detected {len(hit_times)} racket hits")
        return hit_times, y_filtered












import torch 
import torchvision.transforms as transforms 
import torchvision.models as models 
import cv2 


class CourtLineDetector:
    def __init__(self, model_path, machine = 'cpu'):
        

        self.machine = machine
        # Initialize our trained courtline keypoints detection model
        self.model = models.resnet50(pretrained = False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location = self.machine))


        # Transformations for input frames (same as when we trained the model)
        self.transforms = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # Normalize using the mean & std of ImageNet (since we use a
            # ResNet50 model, which was trained on ImageNet)
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        )


    
    def predict(self, frame):
        """
        Predict the keypoints of the tennis court based on our model on a single frame.

        Args:
            frame : input frame.

        Returns:
            keypoints (np.array) : Array of the keypoints.
        
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs[0].cpu().numpy()

        # Convert the keypoints to the correct scale
        og_h, og_w = img_rgb.shape[:2]

        keypoints[::2] *= og_w/224.0
        keypoints[1::2] *= og_h/224.0

    
        return keypoints 

    def draw_keypoints(self, frame, keypoints):
        """
        Draw the keypoints on a single frame.

        Args:
            frame : input frame.
            keypoints : given keypoints.

        Returns:
            frame : the frame annotated with the keypoints
        
        """

        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

          #  cv2.putText(frame, f"{str(i//2)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.circle(frame, (x,y), 2, (0,255,0), -1) # -1 such that it is filled

        return frame 


    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draw Keypoints on all the frames of the video, based on what was 
        found in the frame that was used to detect the keypoints.

        Args:
            video_frames : all frames of the video.
            keypoints : given keypoints.

        Returns:
            output_video_frames : all frames, annotated with the found keypoints
        
        """
        output_video_frames = []

        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            frame = self.draw_lines_between_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        
        return output_video_frames 
    

    def draw_lines_between_keypoints(self, frame, keypoints):
        """
        
        Draw lines between the keypoints 
        
        """

        # For easier indexing of keypoints
        keypoints_zipped = []
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            keypoints_zipped.append((x,y))

        
        # Now connect

        # 0 & 1
        cv2.line(frame, keypoints_zipped[0], keypoints_zipped[1], color=(0,255,0), thickness= 1)
        # 0 & 2
        cv2.line(frame, keypoints_zipped[0], keypoints_zipped[2], color=(0,255,0), thickness= 1)
        # 2 & 3
        cv2.line(frame, keypoints_zipped[2], keypoints_zipped[3], color=(0,255,0), thickness= 1)
        # 1 & 3
        cv2.line(frame, keypoints_zipped[1], keypoints_zipped[3], color=(0,255,0), thickness= 1)
        # 4 & 5
        cv2.line(frame, keypoints_zipped[4], keypoints_zipped[5], color=(0,255,0), thickness= 1)
        # 6 & 7
        cv2.line(frame, keypoints_zipped[6], keypoints_zipped[7], color=(0,255,0), thickness= 1)
        # 10 & 11
        cv2.line(frame, keypoints_zipped[10], keypoints_zipped[11], color=(0,255,0), thickness= 1)
        # 8 & 9 
        cv2.line(frame, keypoints_zipped[8], keypoints_zipped[9], color=(0,255,0), thickness= 1)
        # 12 & 13
        cv2.line(frame, keypoints_zipped[12], keypoints_zipped[13], color=(0,255,0), thickness= 1)


        return frame 




















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
                   draw_player_stats)
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


    DRAW_MINI_COURT = False
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_number = 101
    input_video_path = f'data/input_video{video_number}.mp4'  # Toy example
    #input_video_path = f'data/videos/video_{video_number}.mp4' # Real example

    input_video_path_audio = f'data/input_video{video_number}_audio.mp3'
    convert_mp4_to_mp3(input_video_path, input_video_path_audio)
        
    output_video_path = f'output/output_video{video_number}.mp4'

    # Initialize Tracker for Players & Ball
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')
    ball_tracker_method = 'tracknet' # 'yolo' or 'tracknet'
    
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




    elif ball_tracker_method == 'tracknet':

        # TODO : Put all of this in a class call like above
        # Calculate the correct scale factor for scaling back 
        # with TrackNet, we scaled to 640 width, 360 height

        import pickle
        stub_path = 'tracker_stubs/tracknet_ball_detections.pkl'
        read = True # read from stub
        if stub_path is not None and read == True:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
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
  #  ball_shots_frames = ball_tracker.get_ball_shot_frames(ball_detections)
  #  print(f"Ball shots detected at frames: {ball_shots_frames}")
    # first_player_balls_frames = ball_shots_frames[0::2]
    
    first_player_balls_frames = [52, 113, 177, 235, 305] # Hardcoded for now
    print(f"First player ball shots detected at frames: {first_player_balls_frames}")
    
    # List of frames where ball hits the ground
    ball_landing_frames = [45, 75, 102, 130, 165, 194, 223, 255, 296, 324] # Hardcoded for now

    # Print out the 10 frames before and after each hit (excluding initial and last one)



    print(f"Ball landing frames: {ball_landing_frames}")

    
    # Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                            ball_detections,
                                                                                                                            courtline_keypoints,
                                                                                                                            chosen_players_ids)
    

    # Get racket hits 
    ball_shots_frames_audio = get_ball_shot_frames_audio(input_video_path_audio, fps, plot = False)
    ball_shots_frames = get_ball_shot_frames_visual(ball_mini_court_detections)
    ball_shots_frames = combine_visual_audio(ball_shots_frames, ball_shots_frames_audio, fps)


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
        output_frames = write_track(output_frames, ball_detections_tracknet, trace = 10)
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
   # output_frames = draw_player_stats(output_frames, player_stats_data_df)

    # Draw frame number (top left corner)
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame n {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Save video
    save_video(output_frames, output_video_path, fps)



if __name__ == '__main__':
    main()
