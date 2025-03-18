import numpy as np 
import cv2 
import os 
from copy import deepcopy
import random 
import shutil 
import pandas as pd
from sklearn.cluster import DBSCAN

def cluster_series(arr, eps=3, min_samples=2, delay=2):
    arr = np.array(sorted(map(int, arr))).reshape(-1, 1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(arr)
    
    df = pd.DataFrame({'Value': arr.flatten(), 'Cluster': labels})
    
    # Filter out the outliers (cluster label -1)
    df = df[df['Cluster'] != -1]
    
    min_values = df.groupby('Cluster')['Value'].min().reset_index()
    min_values['Value'] -= delay  # Apply delay adjustment
    
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
            elif any(ball_detect[0] is None for ball_detect in ball_detections[idx:idx+4]):
                continue

            # Else, we run inference on the image
            f = os.path.join(output_path_inference, f"{video_n}_frame_{idx}.jpg")
            cv2.imwrite(f, input_frames[idx])
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
    # First, extend the ball_shots_frames 
    for i in ball_shots_frames_original:
        for j in range(i, i + trace//2):
            ball_shots_frames.append(j)

    # For faster lookup time 
    ball_bounce_frames = set(ball_bounce_frames)
    ball_shots_frames = set(ball_shots_frames)
    bounce_frames_curr = []
    bounce_frames_continuation = []
    for idx,frame in enumerate(input_frames):
        # Exclude the initial ball bounce, as it comes from a serve and the ball tracking there is not optimal
        if idx in ball_bounce_frames and idx > ball_shots_frames_original[1]:
            # These frames have the more characteristic "V" shape that we are looking for
            # Start two frame after the initial bounce (such that a V pattern is more clearly visible)
            # Follow for around 6 frames (this could be done more analytically with the velocity of the
            # ball to understand how much of a v-shape is visible, but for our use case this is enough)

            # NEW LOGIC DEPENDING ON TRACE!!
            
            if trace == 10:
                bounce_frames_curr = [idx+2+i for i in range(6)]
            
            if trace == 3:
                bounce_frames_curr = [idx + 1]

            if trace == 5:
                bounce_frames_curr = [idx+1+i for i in range(3)]


            #### No Bounce Helper
            # Find the next racket hit after a bounce 
            for i in ball_shots_frames_original:
                if i > idx:
                    curr_racket_hit = i 
                    break 
                else:
                    curr_racket_hit = None 


            # This list we create just such that in the "no bounce" examples we don't accidentally include bounce "residues"
            if curr_racket_hit:
                bounce_frames_continuation = [i for i in range(idx, curr_racket_hit)]
            else:
                bounce_frames_continuation = [i for i in range(idx, len(input_frames))]
            #### No Bounce Helper

            # Make sure not to go over bounds 
            bounce_frames_curr = list(filter(lambda x : x <= len(input_frames) - 1, bounce_frames_curr))
            for i in bounce_frames_curr:
                # If already is a racket hit, break
                if i in ball_shots_frames_original:
                    break
                # Furthermore, check that there are atleast 4 consecutive tracks (else we add empty image / too little trace)
                elif any(ball_detect[0] is None for ball_detect in ball_detections[i:i+4]):
                    continue

                f = os.path.join(output_path_bounce, f"{video_n}_frame_{i}.jpg")
                cv2.imwrite(f, input_frames[i])
                # DATA AUGMENTATION : FLIP VERTICALLY (since we have about 2:1 no bounce to bounce ratio)
                f = os.path.join(output_path_bounce, f"{video_n}_frame_flipped_{i}.jpg")
                cv2.imwrite(f, cv2.flip(input_frames[i],1))


        
        elif idx in bounce_frames_continuation or idx in ball_shots_frames:
            continue 
        # no bounce 
        else:
            # Only add examples after second racket hit (bc start is often messy)
            if idx > ball_shots_frames_original[1]:
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

    




    

