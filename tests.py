from ultralytics import YOLO

player_keypoint_tracker_model = YOLO("models/yolo11m-pose.pt")


results = player_keypoint_tracker_model('data/input_video1.mp4', save=True)