from ultralytics import YOLO

player_keypoint_tracker_model = YOLO("models/yolo11m-pose.pt")


results = player_keypoint_tracker_model('data/input_video1.mp4', save=True)




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