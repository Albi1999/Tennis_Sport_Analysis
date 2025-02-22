from ultralytics import YOLO
import cv2
import os

# Set the working directory to the current directory
os.chdir(os.path.dirname(__file__))


# Choose the model to use (i.e. which YOLO model)
model = YOLO('models/yolo11best.pt')

# input data path 
input = 'data/input_video2.mp4'





# Perform prediction (on ball)
model.predict(input,
              conf= 0.10, # low confidence as we only are tracking the one ball (we assume not a lot
                          # of other things in the video will be detected as the ball)
              project = 'output/output_video2.mp4',
              max_det = 1, # only one ball, so we only want to detect one thing at a time
              show = True, # show every frame during processing 
              save=True) # save video



# result = model.predict(...) followed by print(result) to get some useful informations








