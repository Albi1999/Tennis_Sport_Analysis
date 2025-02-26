from ultralytics import YOLO
import supervision as sv
import cv2
import os
import numpy as np

# Set the working directory to the current directory
os.chdir(os.path.dirname(__file__))


# Choose the model to use (i.e. which YOLO model)
model = YOLO('models/yolo11best.pt')

# input data path 
input = 'data/input_frame2.jpg'

# Load the image
image = cv2.imread(input)

# Perform prediction
ball_results = model(image)[0]
player_results = [575.6511840820312, 732.5692749023438, 725.7564697265625, 929.4720458984375]

# Create Detections object from the results
ball_detections = sv.Detections.from_ultralytics(ball_results)
player_detections = sv.Detections(np.array(player_results).reshape(1, 4))
player_detections.class_id = np.array([0])

print(type(ball_detections))
print(type(player_detections))


# Ball Annotators
ball_dot_annotator = sv.DotAnnotator()
ball_dot_annotator.color = sv.Color(255, 255, 0)

ball_triangle_annotator = sv.TriangleAnnotator()
ball_triangle_annotator.color = sv.Color(255, 0, 0)

# Player Annotators
player_ellipse_annotator = sv.EllipseAnnotator()
player_ellipse_annotator.color = sv.Color(0, 0, 255)

player_triangle_annotator = sv.TriangleAnnotator()
player_triangle_annotator.color = sv.Color(0, 0, 255)


# Annotate the image with the ball detections
annotated_frame = ball_dot_annotator.annotate(
    scene=image.copy(),
    detections=ball_detections
)

annotated_frame = ball_triangle_annotator.annotate(
    scene=annotated_frame,
    detections=ball_detections,
)

# Annotate the image with the player detections
annotated_frame = player_ellipse_annotator.annotate(
    scene=annotated_frame,
    detections=player_detections
)

annotated_frame = player_triangle_annotator.annotate(
    scene=annotated_frame,
    detections=player_detections
)


# Download the annotated image
output_path = 'output/output_frame2.jpg'
cv2.imwrite(output_path, annotated_frame)








