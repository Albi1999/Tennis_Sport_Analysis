from ultralytics import YOLO
import cv2
import os

# Set the working directory to the current directory
os.chdir(os.path.dirname(__file__))

model = YOLO('models/yolo11best.pt')
input = 'data/input_video2.mp4'

# Perform prediction
model.predict(input, save=True)








