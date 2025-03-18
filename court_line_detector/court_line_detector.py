import torch
import torchvision.transforms as transforms 
import torchvision.models as models 
import cv2 
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from skimage.measure import ransac


class CourtLineDetector:
    def __init__(self, model_path, machine='cpu'):
        """
        Initialize the CourtLineDetector with the trained model.
        
        Args:
            model_path (str): Path to the trained model.
            machine (str): Device to use ('cpu' or 'cuda').
        """
        self.machine = machine
        
        # Load the trained courtline keypoints detection model
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.machine))
        self.model.eval()
        
        # Transformations for input frames
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Keypoint Tracking
        self.initial_keypoints = None  # Store initial detected keypoints
        self.prev_gray = None  # Previous grayscale frame
        self.prev_keypoints = None  # Previous keypoints
        
        # Reference keypoints for a standard tennis court
        self.reference_court_pts = np.array([
            [100, 100], [500, 100], [100, 400], [500, 400],
            [200, 150], [400, 150], [200, 350], [400, 350],
            [300, 100], [300, 400], [150, 250], [450, 250],
            [250, 250], [350, 250]
        ])
    
    def detect_keypoints(self, frame):
        """
        Detect the keypoints of the tennis court based on the trained model.
        
        Args:
            frame (np.array): Input frame.
        
        Returns:
            np.array: Detected keypoints.
        """
        img_tensor = self.transforms(frame).unsqueeze(0)
        img_tensor = img_tensor.to(self.machine)
        
        with torch.no_grad():
            output = self.model(img_tensor).cpu().numpy().flatten()
        
        keypoints = output.reshape(-1, 2)
        keypoints = self.snap_keypoints_to_template(frame, keypoints)
        return keypoints

    
    def snap_keypoints_to_template(self, frame, keypoints):
        """
        Align detected keypoints to the predefined court template.
        
        Args:
            frame (np.array): Input frame.
            keypoints (np.array): Detected keypoints.
        
        Returns:
            np.array: Aligned keypoints.
        """
        if len(keypoints) != len(self.reference_court_pts):
            return keypoints  # Ensure we have the right number of keypoints
        
        cost_matrix = distance.cdist(keypoints, self.reference_court_pts)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        ordered_keypoints = self.reference_court_pts[col_ind]
        
        return np.array(ordered_keypoints)


    
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
        refined_keypoints = self.refine_keypoints(keypoints)

        # Ensure keypoints are unique
        unique_keypoints = []
        seen = set()
        for kpt in refined_keypoints:
            kpt_tuple = tuple(kpt)
            if kpt_tuple not in seen:
                unique_keypoints.append(kpt)
                seen.add(kpt_tuple)

        return np.array(unique_keypoints, dtype=np.float32)

    def refine_keypoints(self, keypoints):
        """
        Refine keypoints by aligning them to the nearest line intersection.

        Args:
            keypoints (list): Keypoints.
            
        Returns:
            refined_kps (array): Array of refined keypoints.
        """
        refined_kps = []
        for kp in keypoints:
            close_kps = [p for p in keypoints if distance.euclidean(kp, p) < 20]
            avg_kp = np.mean(close_kps, axis=0) if close_kps else kp
            refined_kps.append(avg_kp)
        return np.array(refined_kps)



    def draw_keypoints(self, frame, refined_keypoints):
        """
        Draw the keypoints on a single frame.

        Args:
            frame : input frame.
            keypoints : given keypoints.

        Returns:
            frame : the frame annotated with the keypoints
        """
        for i, (x, y) in enumerate(refined_keypoints): # Draw the refined keypoints
            x, y = int(x), int(y) 
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) # Draw the keypoint
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
            frame = self.draw_lines_between_keypoints(frame, refined_keypoints)
            output_video_frames.append(frame)
        
        return output_video_frames 
    


    def draw_lines_between_keypoints(self, frame, keypoints):
        """
        Draw lines between the keypoints 

        Args:
            frame : input frame.
            keypoints : given keypoints.

        Returns:
            frame : the frame annotated with the lines between the keypoints
        """
        # Convert keypoints into tuples of integers
        keypoints = keypoints.astype(np.int32).tolist()

        # Convert to list of tuples
        keypoints = list(map(tuple, keypoints))


        # 0 & 1
        cv2.line(frame, keypoints[0], keypoints[1], color=(0,255,0), thickness= 1)
        # 0 & 2
        cv2.line(frame, keypoints[0], keypoints[2], color=(0,255,0), thickness= 1)
        # 2 & 3
        cv2.line(frame, keypoints[2], keypoints[3], color=(0,255,0), thickness= 1)
        # 1 & 3
        cv2.line(frame, keypoints[1], keypoints[3], color=(0,255,0), thickness= 1)
        # 4 & 5
        cv2.line(frame, keypoints[4], keypoints[5], color=(0,255,0), thickness= 1)
        # 6 & 7
        cv2.line(frame, keypoints[6], keypoints[7], color=(0,255,0), thickness= 1)
        # 10 & 11
        cv2.line(frame, keypoints[10], keypoints[11], color=(0,255,0), thickness= 1)
        # 8 & 9 
        cv2.line(frame, keypoints[8], keypoints[9], color=(0,255,0), thickness= 1)
        # 12 & 13
        cv2.line(frame, keypoints[12], keypoints[13], color=(0,255,0), thickness= 1)

        return frame 

