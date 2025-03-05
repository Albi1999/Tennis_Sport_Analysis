import numpy as np 
import cv2 
import os 
from copy import deepcopy
import random 
import shutil 

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

    if not os.path.exists(output_path):
        os.makedirs(output_path)
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



def scraping_data(video_n, output_path, input_frames, ball_bounce_frames, ball_shots_frames, trace):

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
    # First, extend the ball_shots_frames 
    for i in ball_shots_frames_original:
        for j in range(i, i + trace//2):
            ball_shots_frames.append(j)

    # For faster lookup time 
    ball_bounce_frames = set(ball_bounce_frames)
    ball_shots_frames = set(ball_shots_frames)
    bounce_frames_curr = []
    for idx,frame in enumerate(input_frames):
        if idx in ball_bounce_frames:
            # These frames have the more characteristic "V" shape that we are looking for
            bounce_frames_curr = [idx+i for i in range(trace - 1)]
            for i in bounce_frames_curr:
                # If already is a racket hit, break
                if i in ball_shots_frames_original:
                    break
                f = os.path.join(output_path_bounce, f"{video_n}_frame_{i}.jpg")
                cv2.imwrite(f, input_frames[i])


        # TODO : maybe even extend bounce_frames_curr by like 2 frames before, because that is where the ball already falls into the more v shape thing
        if idx in bounce_frames_curr or idx in ball_shots_frames:
            continue 
        # no bounce 
        else:
            f = os.path.join(output_path_no_bounce, f"{video_n}_frame_{idx}.jpg")
            cv2.imwrite(f, frame)



def splitting_data(main_dir, train_ratio = 0.75, val_ratio = 0.15, test_ratio = 0.10):
    


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
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split sizes - stratified by class
        n_files = len(image_files)
        n_train = int(train_ratio * n_files)
        n_val = int(val_ratio * n_files)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Record distribution for this class
        distribution[class_name] = {
            'total': n_files,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'label': label
        }
        
        # Copy files to respective directories, maintaining class naming
        target_class_name = class_name  # Using original class name instead of "class_0", "class_1"
        
        for files, target_set in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            target_dir = os.path.join(main_dir, target_set, target_class_name)
            for file_name in files:
                src = os.path.join(class_source, file_name)
                dst = os.path.join(target_dir, file_name)
                shutil.copy2(src, dst)
    
    # Print distribution summary
    print(f"Dataset split complete with stratification. Distribution summary:")
    for class_name, stats in distribution.items():
        print(f"  {class_name} (Label: {stats['label']}): Total={stats['total']}, "
              f"Train={stats['train']} ({stats['train']/stats['total']:.1%}), "
              f"Val={stats['val']} ({stats['val']/stats['total']:.1%}), "
              f"Test={stats['test']} ({stats['test']/stats['total']:.1%})")
    
    return train_dir, val_dir, test_dir

    




    

