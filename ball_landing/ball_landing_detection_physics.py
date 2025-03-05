from utils import get_center_of_bbox,convert_pixel_distance_to_meters
import math
import numpy as np
import pandas as pd 
import sys 
import info 
sys.path.append('../')

class BallLandingDetector:
    """
    Main idea : We have problems with estimating the vertical velocity due to the angle of the video. Instead, we try to
    focus more on finding the time it takes for the ball to land : We therefore don't try to calculate the trajectory of
    the ball, but rather approximate the time it takes the ball to land on the ground.
    This class analyzes a single shot, from racket hit to ball landing.
    
    
    """
    def __init__(self, video_frames, racket_hit_frames, idx, fps, ball_positions, reference, shot_type = 'normal'):
        """
        
        Args :
            shot_type : Type of shot (normal or serve)
            ball_positions : of TrackNet (NOT the ones of the mini court!)
        
        """
        self.video_frames = video_frames 
        self.idx = idx 
        self.racket_hit_frames = racket_hit_frames
        self.racket_hit_frame = racket_hit_frames[idx] # The actual hit that we're interested in 
        self.fps = fps 
        self.ball_positions = ball_positions
        self.reference = reference
        self.shot_type = shot_type 


    def detect_highest_point(self):
        """
        Detect the (approximated) highest point of the ball according to the y coordinates
        that we have of each given frame. Note that this is not exact due to the perspective
        skew (i.e. the camera angle that we have doesn't represent the actual vertical
        position).
        """

        # Look for the highest point between two consecutive racket hits or between the last racket hit & end of video

        accessible_indices = [x for x in range(len(self.racket_hit_frames))]
        # Check if there is another racket hit following
        if (self.idx + 1) in accessible_indices:
            frame_range = [x for x in range(self.racket_hit_frame, self.racket_hit_frames[self.idx + 1])]
        else: 
            frame_range = [x for x in range(self.racket_hit_frame, len(self.video_frames))]

        
        # Get the centered y values of the ball positions of the frames we are currently analyzing
        y_positions = []
        for frame in frame_range:
            # Get the bbox coordinates 
            bbox = list(self.ball_positions[frame].values())[0]
            # Get center of bboxes (such that this can be run with either YOLO or TrackNet detections)
            _ ,y = get_center_of_bbox(bbox)
            y_positions.append(y)

        
        # Get the rolling mean
        window_size = 5
        rolling_mean_y_positions = pd.Series(y_positions).rolling(window=window_size).mean()
        rolling_mean_y_positions = rolling_mean_y_positions.fillna(method = 'bfill')
        y_positions_rolled = rolling_mean_y_positions.tolist()
        # Get both the frame & the actual highest point
        frame_highest_point = frame_range[np.argmax(y_positions_rolled)]
        height_highest_point = int(y_positions[np.argmax(y_positions_rolled)])


        return frame_highest_point, height_highest_point
    


    def detect_t0(self,frame_highest_point):
        """ 
        Frames it takes from the initial hit to the highest point in the ball's trajectory
        """

        return frame_highest_point - self.racket_hit_frame
    

    def detect_t1(self, height_highest_point):
        """
        Time it takes from the highest point to the floor. Governed by the formula for kinematic equation of vertical motion, i.e. 

        height = v_vert_initial * t + 1/2 * g * t^2 . Since v_vert_initial = 0 at highest point, we just have
        height = 1/2 * g * t^2, and therefore, in our use case
        t = sqrt(2 * (height_highest_point/g))
        
        """
        
        g = 9.81
        height_highest_point_meters = convert_pixel_distance_to_meters(height_highest_point, info.DOUBLE_LINE_WIDTH, self.reference)
        fall_time = math.sqrt(2 * (height_highest_point_meters/g))

        return self.__time_to_frames(fall_time)
    


    def time_ball_trajectory(self):
        """
        Calculate the full time it takes from an initial racket hit to ball landing
        and convert the value in frames. Furthermore, get the approximated frame
        of the ball landing on the floor.
        
        """

        frame_highest_point, height_highest_point = self.detect_highest_point()
        t0 = self.detect_t0(frame_highest_point)
        t1 = self.detect_t1(height_highest_point)
        full_frames = t0 + t1
        ball_landing_frame = self.racket_hit_frame + full_frames



        return full_frames, ball_landing_frame
    

    def correction_by_averaging(self):
        """
        Try to correct the actual ball landing frame by look at the 10 frames before
        and after the approximated ball landing and averaging (x,y) coordiantes.
        Check if this is needed 
        """


        corrected_frame = None 
        return corrected_frame 
    



    def __time_to_frames(self, time):
        """
        Convert time to frames using fps of our video. 
        """
        frames = time * self.fps 

        return frames



