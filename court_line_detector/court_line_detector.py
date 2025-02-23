import torch 
import torchvision.transforms as transforms 
import torchvision.models as models 
import cv2 


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

    
        return keypoints 

    def draw_keypoints(self, frame, keypoints):
        """
        Draw the keypoints on a single frame.

        Args:
            frame : input frame.
            keypoints : given keypoints.

        Returns:
            frame : the frame annotated with the keypoints
        
        """

        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(frame, f"K {str(i//2)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.circle(frame, (x,y), 5, (0,255,0), -1) # -1 such that it is filled

        return frame 


    def draw_keypoints_on_video(self, video_frames, keypoints):
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
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        
        return output_video_frames 

