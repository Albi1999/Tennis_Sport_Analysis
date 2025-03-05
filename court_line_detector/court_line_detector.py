import torch 
import torchvision.transforms as transforms 
import torchvision.models as models 
import cv2 
import numpy as np
from scipy.spatial import distance


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

        keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

        return self.postprocess_keypoints(frame, keypoints)
    
        
    def postprocess_keypoints(self, frame, keypoints):
        """
        Postprocess the keypoints by refining them using detected lines and intersections.

        Args:
            frame (np.array): Input video frame.
            keypoints (np.array): Raw keypoints from the model.

        Returns:
            refined_keypoints (np.array): Post-processed keypoints.
        """
        refined_keypoints = self.refine_keypoints(frame, keypoints)

        # Ensure keypoints are unique
        unique_keypoints = []
        seen = set()
        for kpt in refined_keypoints:
            kpt_tuple = tuple(kpt)
            if kpt_tuple not in seen:
                unique_keypoints.append(kpt)
                seen.add(kpt_tuple)

        return np.array(unique_keypoints, dtype=np.float32)

    def refine_keypoints(self, image, keypoints):
        """
        Refine keypoints by aligning them to the nearest line intersection.

        Args:
            image (np.array): Input image.
            keypoints (list): Initial keypoints.
            
        Returns:
            refined_keypoints (list): List of refined keypoints.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = self.detect_lines(gray)
        merged_lines = self.merge_lines(lines)
        intersections = self.find_intersections(merged_lines)

        refined_keypoints = []
        for x, y in keypoints:
            if intersections:
                dists = [distance.euclidean((x, y), inter) for inter in intersections]
                closest_inter = intersections[np.argmin(dists)]
                refined_keypoints.append(closest_inter)
            else:
                refined_keypoints.append((x, y))  # Fallback if no intersections found

        return np.array(refined_keypoints, dtype=np.float32)

    def detect_lines(self, image):
        """
        Detect lines in the image using Hough Transform.
        
        Args:
            image (np.array): Input grayscale image.
            
        Returns:
            lines (list): Detected lines.
        """
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)
        return lines if lines is not None else []

    def merge_lines(self, lines):
        """
        Merge similar lines based on proximity.

        Args:
            lines (list): List of detected lines.
            
        Returns:
            merged_lines (list): Merged lines.
        """
        if len(lines) == 0:
            return []

        merged_lines = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            x1, y1, x2, y2 = line1[0]
            merged = [x1, y1, x2, y2]
            count = 1
            
            for j, line2 in enumerate(lines):
                if i != j and j not in used:
                    x3, y3, x4, y4 = line2[0]
                    if distance.euclidean((x1, y1), (x3, y3)) < 10 and distance.euclidean((x2, y2), (x4, y4)) < 10:
                        merged[0] += x3
                        merged[1] += y3
                        merged[2] += x4
                        merged[3] += y4
                        count += 1
                        used.add(j)
            
            merged = [int(v / count) for v in merged]
            merged_lines.append(merged)
        
        return merged_lines

    def find_intersections(self, lines):
        """
        Find intersections between detected lines.

        Args:
            lines (list): List of detected lines.
            
        Returns:
            intersections (list): List of intersection points.
        """
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if i >= j:
                    continue
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    continue  # Parallel lines

                px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

                if 0 <= px <= 640 and 0 <= py <= 480:  # Assuming the frame size
                    intersections.append((int(px), int(py)))

        return intersections

    def draw_keypoints(self, frame, refined_keypoints):
        """
        Draw the keypoints on a single frame.

        Args:
            frame : input frame.
            keypoints : given keypoints.

        Returns:
            frame : the frame annotated with the keypoints
        
        """
        # Precedent function to draw keypoints on the frame
        #for i in range(0, len(keypoints), 2):
        #    x = int(keypoints[i])
        #    y = int(keypoints[i+1])

        #    cv2.putText(frame, f"K {str(i//2)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        #    cv2.circle(frame, (x,y), 5, (0,255,0), -1) # -1 such that it is filled

        #return frame 
        for i, (x, y) in enumerate(refined_keypoints): # Draw the refined keypoints
            x, y = int(x), int(y) 
            cv2.putText(frame, f"K {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Put the keypoint number
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Draw the keypoint
        return frame

    def draw_keypoints_on_video(self, video_frames, refined_keypoints):
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
            frame = self.draw_keypoints(frame, refined_keypoints)
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

