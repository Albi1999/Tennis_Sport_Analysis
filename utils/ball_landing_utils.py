import numpy as np 
import cv2 
import os 
from copy import deepcopy
import random 
import shutil 
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import defaultdict
import re


def cluster_series(arr, eps=3, min_samples=2, delay=2):
    """Cluster a series of frames and return the minimum value of each cluster with a delay adjustment."""
    if len(arr) == 0:
        print("No ball landing frames detected.")
        return []
    
    else:
        arr = np.array(sorted(map(int, arr))).reshape(-1, 1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(arr)
        
        df = pd.DataFrame({'Value': arr.flatten(), 'Cluster': labels})
        
        # Filter out the outliers (cluster label -1)
        df = df[df['Cluster'] != -1]
        
        min_values = df.groupby('Cluster')['Value'].min().reset_index()
        min_values['Value'] -= delay  # Apply delay adjustment


      #  final_values = []
      #  for i in min_values.values.tolist():
      #      final_values.append(i[1])


        return min_values['Value'].values.tolist()
        

def create_black_video(output_path, width, height, fps, frame_count):
    """
    Create a black video with specified parameters.
    
    Args:
        output_path: Path to save the output video
        width: Width of the video in pixels
        height: Height of the video in pixels
        fps: Frames per second
        frame_count: Total number of frames
    """

  #  if not os.path.exists(output_path):
  #      os.makedirs(output_path)
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a black frame
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Write the black frame repeatedly
    for _ in range(frame_count):
        out.write(black_frame)
    
    # Release the video writer
    out.release()


def scraping_data_for_inference(video_n, output_path, input_frames, ball_shots_frames, trace , ball_detections):

    output_path_inference = output_path + '/images_inference'


    # Delete the folder if it exists
    if os.path.exists(output_path_inference):
        # Remove all files in the directory
        for file in os.listdir(output_path_inference):
            file_path = os.path.join(output_path_inference, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)


    if not os.path.exists(output_path_inference):
        os.makedirs(output_path_inference)


    # Extend ball_shots_frames 
    ball_shots_frames_original = deepcopy(ball_shots_frames)
    # First, extend the ball_shots_frames 
    for i in ball_shots_frames_original:
        for j in range(i, i + trace//2):
            ball_shots_frames.append(j)

    ball_shots_frames = set(ball_shots_frames)

    saved_frames = []
    for idx,frame in enumerate(input_frames):
        
        # We don't look at the initial hit
        if idx > ball_shots_frames_original[1]:
    
            # If it is a racket hit (or directly after a racket hit)
            if idx in ball_shots_frames:
                continue
            # Furthermore, check that there are atleast 4 consecutive tracks (else we add empty image / too little trace)
            elif all(ball_detect[0] is None for ball_detect in ball_detections[idx:idx+4]):
                continue

            # Else, we run inference on the image
            f = os.path.join(output_path_inference, f"{video_n}_frame_{idx}.jpg")
            cv2.imwrite(f, cv2.cvtColor(input_frames[idx], cv2.COLOR_BGR2GRAY))
            saved_frames.append(idx)


    return saved_frames




def scraping_data(video_n, output_path, input_frames, ball_bounce_frames, ball_shots_frames, trace, ball_detections):
    """
    Scrape the training data 
    """
    output_path_bounce = output_path + '/bounce'
    output_path_no_bounce = output_path + '/no_bounce'

    if not os.path.exists(output_path_bounce):
        os.makedirs(output_path_bounce)

    if not os.path.exists(output_path_no_bounce):
        os.makedirs(output_path_no_bounce)

    # Extend ball_shots_frames 
    ball_shots_frames_original = deepcopy(ball_shots_frames)

    # Extend the ball_shots_frames 
    for i in ball_shots_frames_original:
        for j in range(i, i + trace//2):
            ball_shots_frames.append(j)

    # Convert sets for faster lookup time 
    ball_bounce_frames = set(ball_bounce_frames)
    ball_shots_frames = set(ball_shots_frames)
    bounce_frames_curr = []
    bounce_frames_continuation = []

    for idx, frame in enumerate(input_frames):
        # Convert each frame to greyscale before processing or saving
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if idx in ball_bounce_frames and idx > ball_shots_frames_original[1]:
            if trace == 10:
                bounce_frames_curr = [idx+2+i for i in range(6)]
            elif trace == 3:
                bounce_frames_curr = [idx + 1]
            elif trace == 6:
                bounce_frames_curr = [idx+2+i for i in range(3)]

            # Find the next racket hit after a bounce
            for i in ball_shots_frames_original:
                if i > idx:
                    curr_racket_hit = i 
                    break 
                else:
                    curr_racket_hit = None

            if curr_racket_hit:
                bounce_frames_continuation = [i for i in range(idx, curr_racket_hit)]
            else:
                bounce_frames_continuation = [i for i in range(idx, len(input_frames))]

            bounce_frames_curr = list(filter(lambda x: x <= len(input_frames) - 1, bounce_frames_curr))

            for i in bounce_frames_curr:
                if i in ball_shots_frames_original:
                    break
                # trained on any, but should try to switch to all!
                elif any(ball_detect[0] is None for ball_detect in ball_detections[i:i+4]):
                    continue

                # Save the greyscale frame
                curr_frame_gray = cv2.cvtColor(input_frames[i], cv2.COLOR_BGR2GRAY)
                f = os.path.join(output_path_bounce, f"{video_n}_frame_{i}.jpg")
                cv2.imwrite(f, curr_frame_gray)
                
                # Save the flipped version (also in greyscale)
                f = os.path.join(output_path_bounce, f"{video_n}_frame_flipped_{i}.jpg")
                cv2.imwrite(f, cv2.flip(curr_frame_gray, 1))

        elif idx in bounce_frames_continuation or idx in ball_shots_frames:
            continue 
        else:
            if idx > ball_shots_frames_original[1]:
                f = os.path.join(output_path_no_bounce, f"{video_n}_frame_{idx}.jpg")
                cv2.imwrite(f, frame_gray)



def splitting_data(main_dir, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directories
    train_dir = os.path.join(main_dir, "train")
    val_dir = os.path.join(main_dir, "val")
    test_dir = os.path.join(main_dir, "test")
    
    # Create class directories for each split
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, "no_bounce"), exist_ok=True)  # class 0
        os.makedirs(os.path.join(dir_path, "bounce"), exist_ok=True)     # class 1

    # Map class names to labels
    class_label_map = {
        "no_bounce": 0,
        "bounce": 1
    }
    
    # Distribution dictionary for reporting
    distribution = {}
    
    # Process each class separately to ensure stratification
    for class_name, label in class_label_map.items():
        class_source = os.path.join(main_dir, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_source) 
                      if f.lower().endswith(('.jpg'))]
        
        # Group files by video number
        videos = defaultdict(list)
        for file in image_files:
            # Extract video number from filename (e.g., "100" from "100_frame_...")
            try:
                video_num = int(file.split('_')[0])
                videos[video_num].append(file)
            except (IndexError, ValueError):
                print(f"Warning: Couldn't extract video number from {file}, skipping...")
                continue
        
        # Get sorted list of video numbers
        video_numbers = sorted(videos.keys())
        
        # Calculate total frames
        total_frames = len(image_files)
        target_train_frames = int(total_frames * train_ratio)
        target_val_frames = int(total_frames * val_ratio)
        
        # Initialize counters
        train_frames = 0
        val_frames = 0
        test_frames = 0
        
        # Initialize video lists for each split
        train_videos = []
        val_videos = []
        test_videos = []
        
        # Sequential assignment of videos to sets
        for video_num in video_numbers:
            video_frame_count = len(videos[video_num])
            
            # Determine which set to assign this video to
            if train_frames < target_train_frames or (not train_videos and not val_videos and not test_videos):
                # Assign to train if below target or if it's the first video
                train_videos.append(video_num)
                train_frames += video_frame_count
            elif val_frames < target_val_frames:
                # Assign to validation if below target
                val_videos.append(video_num)
                val_frames += video_frame_count
            else:
                # Assign remaining to test
                test_videos.append(video_num)
                test_frames += video_frame_count
        
        # Gather files for each split
        train_files = [f for v in train_videos for f in videos[v]]
        val_files = [f for v in val_videos for f in videos[v]]
        test_files = [f for v in test_videos for f in videos[v]]
        
        # Record distribution for this class
        distribution[class_name] = {
            'total': total_frames,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'train_videos': train_videos,
            'val_videos': val_videos,
            'test_videos': test_videos,
            'label': label
        }
        
        # Copy files to respective directories
        for files, target_set in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            target_dir = os.path.join(main_dir, target_set, class_name)
            for file_name in files:
                src = os.path.join(class_source, file_name)
                dst = os.path.join(target_dir, file_name)
                shutil.copy2(src, dst)
    
    # Print distribution summary
    print(f"Dataset split complete with stratification by video. Distribution summary:")
    for class_name, stats in distribution.items():
        print(f"  {class_name} (Label: {stats['label']}): Total={stats['total']}, "
              f"Train={stats['train']} ({stats['train']/stats['total']:.1%}), "
              f"Val={stats['val']} ({stats['val']/stats['total']:.1%}), "
              f"Test={stats['test']} ({stats['test']/stats['total']:.1%})")
        
        print(f"    Train videos: {sorted(stats['train_videos'])}")
        print(f"    Val videos: {sorted(stats['val_videos'])}")
        print(f"    Test videos: {sorted(stats['test_videos'])}")
    
    return train_dir, val_dir, test_dir


def splitting_data_by_video(main_dir, train_videos, val_videos, test_videos):
    """
    Split data by manually specifying which video numbers go into which split.
    
    Args:
        main_dir: Directory containing 'bounce' and 'no_bounce' folders
        train_videos: List of video numbers to include in training set
        val_videos: List of video numbers to include in validation set
        test_videos: List of video numbers to include in test set
    
    Returns:
        Paths to train, validation, and test directories
    """
    # Verify no overlap between splits
    assert len(set(train_videos) & set(val_videos)) == 0, "Train and validation videos overlap"
    assert len(set(train_videos) & set(test_videos)) == 0, "Train and test videos overlap"
    assert len(set(val_videos) & set(test_videos)) == 0, "Validation and test videos overlap"
    
    # Create output directories
    train_dir = os.path.join(main_dir, "train")
    val_dir = os.path.join(main_dir, "val")
    test_dir = os.path.join(main_dir, "test")
    
    # Create class directories for each split
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, "no_bounce"), exist_ok=True)  # class 0
        os.makedirs(os.path.join(dir_path, "bounce"), exist_ok=True)     # class 1

    # Map class names to labels
    class_label_map = {
        "no_bounce": 0,
        "bounce": 1
    }
    
    # Map video numbers to the respective splits
    split_mapping = {}
    for video in train_videos:
        split_mapping[video] = 'train'
    for video in val_videos:
        split_mapping[video] = 'val'
    for video in test_videos:
        split_mapping[video] = 'test'
    
    # Function to extract video number from filename
    def extract_video_number(filename):
        match = re.match(r'(\d+)_frame', filename)
        if match:
            return int(match.group(1))
        return None
    
    # Group files by video number for each class
    video_groups = {class_name: defaultdict(list) for class_name in class_label_map.keys()}
    
    # Distribution dictionary for reporting
    distribution = {class_name: {'total': 0, 'train': 0, 'val': 0, 'test': 0, 'label': label} 
                    for class_name, label in class_label_map.items()}
    
    # Process each class
    for class_name in class_label_map.keys():
        class_source = os.path.join(main_dir, class_name)
        
        # Make sure the source directory exists
        if not os.path.exists(class_source):
            print(f"Warning: Source directory {class_source} doesn't exist. Skipping.")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(class_source) 
                      if f.lower().endswith(('.jpg'))]
        
        # Group by video number
        unassigned_videos = set()
        for file_name in image_files:
            video_number = extract_video_number(file_name)
            if video_number is not None:
                video_groups[class_name][video_number].append(file_name)
                if video_number not in split_mapping:
                    unassigned_videos.add(video_number)
        
        # Warn about unassigned videos
        if unassigned_videos:
            print(f"Warning: The following videos in {class_name} are not assigned to any split: {sorted(unassigned_videos)}")
        
        # Copy files and update distribution
        for video_num, files in video_groups[class_name].items():
            # Skip if video not in any split
            if video_num not in split_mapping:
                continue
                
            split_name = split_mapping[video_num]
            target_dir = os.path.join(main_dir, split_name, class_name)
            
            for file_name in files:
                src = os.path.join(class_source, file_name)
                dst = os.path.join(target_dir, file_name)
                shutil.copy2(src, dst)
            
            # Update distribution stats
            file_count = len(files)
            distribution[class_name]['total'] += file_count
            distribution[class_name][split_name] += file_count
    
    # Print distribution summary
    print(f"Dataset split complete with custom video assignments. Distribution summary:")
    for class_name, stats in distribution.items():
        if stats['total'] > 0:
            print(f"  {class_name} (Label: {stats['label']}): Total={stats['total']}, "
                  f"Train={stats['train']} ({stats['train']/stats['total']:.1%}), "
                  f"Val={stats['val']} ({stats['val']/stats['total']:.1%}), "
                  f"Test={stats['test']} ({stats['test']/stats['total']:.1%})")
        else:
            print(f"  {class_name} (Label: {stats['label']}): No files processed")
    
    # Print video assignments
    print("\nVideo assignments:")
    for class_name in class_label_map.keys():
        videos_by_split = {'train': [], 'val': [], 'test': []}
        for video_num in video_groups[class_name]:
            if video_num in split_mapping:
                videos_by_split[split_mapping[video_num]].append(video_num)
        
        print(f"\n{class_name}:")
        for split, videos in videos_by_split.items():
            videos_str = ', '.join(map(str, sorted(videos)))
            print(f"  {split}: {videos_str}")
    
    return train_dir, val_dir, test_dir


    


def draw_ball_landings(video_frames, ball_landing_frames, ground_truth_bounce, ball_detections, K=10):
    """
    Draws ball bounces on the video, keeping them visible for K consecutive frames:
    - Predicted bounces in red
    - Ground truth bounces in green
    
    Args:
        video_frames: List of video frames
        ball_landing_frames: List of frames where the ball bounces (predictions)
        ground_truth_bounce: List of frames where the ball bounces (ground truth)
        ball_detections: Ball coordinates for each frame
        K: Number of consecutive frames to display each bounce
    
    Returns:
        List of video frames with bounces drawn
    """
    # Create a copy of the video frames
    output_frames = video_frames.copy()
    
    # Prepare a dictionary for each frame, containing the bounces to be drawn
    predicted_bounces_to_draw = {}  # {frame_idx: [bounce_idx1, bounce_idx2, ...]}
    ground_truth_to_draw = {}       # {frame_idx: [bounce_idx1, bounce_idx2, ...]}
    
    # Populate the dictionary for predicted bounces
    for bounce_idx in ball_landing_frames:
        # For each bounce, add its index to the next K frames
        for k in range(K):
            frame_to_update = bounce_idx + k
            if frame_to_update < len(output_frames):
                if frame_to_update not in predicted_bounces_to_draw:
                    predicted_bounces_to_draw[frame_to_update] = []
                predicted_bounces_to_draw[frame_to_update].append(bounce_idx)
    
    # Populate the dictionary for ground truth bounces
    for bounce_idx in ground_truth_bounce:
        # For each bounce, add its index to the next K frames
        for k in range(K):
            frame_to_update = bounce_idx + k
            if frame_to_update < len(output_frames):
                if frame_to_update not in ground_truth_to_draw:
                    ground_truth_to_draw[frame_to_update] = []
                ground_truth_to_draw[frame_to_update].append(bounce_idx)
    
    # Iterate through each video frame
    for idx, frame in enumerate(output_frames):
        # Draw legend in the top left corner
        cv2.putText(frame, "Bounce Ground Truth", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Bounce Prediction", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
        # Draw ground truth bounces for this frame
        if idx in ground_truth_to_draw:
            for bounce_idx in ground_truth_to_draw[idx]:
                if bounce_idx < len(ball_detections) and ball_detections[bounce_idx][0] is not None:
                    x, y = int(ball_detections[bounce_idx][0]), int(ball_detections[bounce_idx][1])
                    cv2.circle(frame, (x, y), radius=15, color=(0, 255, 0), thickness=3)
                    cv2.putText(frame, f"n. {bounce_idx}", (x - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw predicted bounces for this frame
        if idx in predicted_bounces_to_draw:
            for bounce_idx in predicted_bounces_to_draw[idx]:
                if bounce_idx < len(ball_detections) and ball_detections[bounce_idx][0] is not None:
                    x, y = int(ball_detections[bounce_idx][0]), int(ball_detections[bounce_idx][1])
                    cv2.circle(frame, (x, y), radius=15, color=(0, 0, 255), thickness=3)
                    cv2.putText(frame, f"n. {bounce_idx}", (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output_frames

