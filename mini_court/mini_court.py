import cv2
import os
import numpy as np
import sys
sys.path.append('../')
import info
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_center_of_bbox,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    euclidean_distance,
    measure_xy_distance,
    get_center_of_bbox
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250          # Width of the mini court in pixels (depends on the image size)
        self.drawing_rectangle_height = 500         # Height of the mini court in pixels (depends on the image size)
        self.buffer = 50                            # Distance from the right and top of the frame to the mini court
        self.padding_court= 50                      # Padding from the mini court to the court (depends on the image size)

        # Define the mini court in the frame
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        
        
    def convert_meters_to_pixels(self, meters):
        '''Convert meters to pixels based on the width of the mini court'''
        return convert_meters_to_pixel_distance(meters,
                                                info.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )
    
    def set_canvas_background_box_position(self,frame):
        '''Set the position of the background rectangle in the frame'''
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer                       # x coordinate of the bottom right corner of the rectangle in the frame
        self.end_y = self.buffer + self.drawing_rectangle_height        # y coordinate of the bottom right corner of the rectangle in the frame
        self.start_x = self.end_x - self.drawing_rectangle_width        # x coordinate of the top left corner of the rectangle in the frame
        self.start_y = self.end_y - self.drawing_rectangle_height       # y coordinate of the top left corner of the rectangle in the frame
        
    def set_mini_court_position(self):
        '''Set the position of the mini court in the frame'''
        self.court_start_x = self.start_x + self.padding_court          # x coordinate of the top left corner of the mini court in the frame
        self.court_start_y = self.start_y + self.padding_court          # y coordinate of the top left corner of the mini court in the frame
        self.court_end_x = self.end_x - self.padding_court              # x coordinate of the bottom right corner of the mini court in the frame
        self.court_end_y = self.end_y - self.padding_court              # y coordinate of the bottom right corner of the mini court in the frame
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # Court Key Points
        # point 0 --> top left corner
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1 --> top right corner
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2 --> bottom left corner
        drawing_key_points[4] = int(self.court_start_x) 
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(info.HALF_COURT_LINE_HEIGHT*2)
        # point 3 --> bottom right corner
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        
        ## Double Ally Key and No Mans Land Points
        # #point 4 --> top left corner of the double ally
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(info.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5 --> top right corner of the double ally
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(info.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6 --> bottom left corner of the double ally
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(info.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7 --> bottom right corner of the double ally
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(info.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8 --> top left corner of the no mans land
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(info.NO_MANS_LAND_HEIGHT)
        # # #point 9 --> top right corner of the no mans land
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(info.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10 --> bottom left corner of the no mans land
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(info.NO_MANS_LAND_HEIGHT)
        # # #point 11 --> bottom right corner of the no mans land
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(info.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12 --> middle of the no mans land
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13 --> middle of the no mans land
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        '''Set the lines of the mini court connecting the key points'''
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),      
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def draw_court(self, frame):
        '''Draw the mini court on the frame'''
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            # Keypoints (Green)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Court lines (White)
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

        # Net (Purple)
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 255), 2)

        return frame

    def draw_background_rectangle(self, frame):
        '''Draw the background rectangle on the frame'''
        out = frame.copy()
        # Rectangle (Black)
        cv2.rectangle(out, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 0), cv2.FILLED)
        return out

    def draw_mini_court(self,frames):
        '''Draw the mini court on all the frames'''
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        """ Convert the position of the player to the mini court coordinates.
        
        Args:
            object_position (tuple): Position of the player in the original frame.
            closest_key_point (tuple): Closest key point to the player in the original frame.
            closest_key_point_index (int): Index of the closest key point to the player.
            player_height_in_pixels (int): Height of the player in pixels.
            player_height_in_meters (int): Height of the player in meters.
        
        Returns:
            mini_court_player_position (tuple): Position of the player in the mini court coordinates.
        """
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position


    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points, chosen_players_ids):
        """
        Convert the bounding boxes of the players and the ball to mini court coordinates 
        using homography transformation.
        
        Args:
            player_boxes (list): List of dictionaries containing the bounding boxes of the players.
            ball_boxes (list): List of dictionaries containing the bounding boxes of the ball.
            original_court_key_points (list): List of the key points of the court in the original frame.
            chosen_players_ids (list): List of the IDs of the chosen players.
            
        Returns:
            output_player_boxes (list): List of dictionaries containing player positions in mini court.
            output_ball_boxes (list): List of dictionaries containing ball positions in mini court.
        """
        output_player_boxes = []
        output_ball_boxes = []
        
        # Create source and destination points for homography
        #src_points = []
        #dst_points = []
        
        # Use court keypoints to create source and destination points for homography
        # We'll use the 4 corners of the court plus additional key points if available
        #key_point_indices = [0, 1, 2, 3]  # Corners of the court
        
        #for idx in key_point_indices:
        #    # Source points from the original court
        #    src_points.append([original_court_key_points[idx*2], original_court_key_points[idx*2+1]])
            
            # Destination points in the mini court
        #    dst_points.append([self.drawing_key_points[idx*2], self.drawing_key_points[idx*2+1]])
        
        # Add more keypoints if we have them (like service lines, etc.)
        #for idx in [4, 5, 6, 7, 12, 13]:
        #    if idx*2+1 < len(original_court_key_points) and idx*2+1 < len(self.drawing_key_points):
        #        src_points.append([original_court_key_points[idx*2], original_court_key_points[idx*2+1]])
        #        dst_points.append([self.drawing_key_points[idx*2], self.drawing_key_points[idx*2+1]])
        
         # Ensure we use all 14 keypoints
        key_point_indices = list(range(14))  # Get all 14 keypoints
        src_points = []
        dst_points = []

        # Modified code to handle the actual structure of original_court_key_points
        for idx in range(min(14, len(original_court_key_points))):
            src_points.append(original_court_key_points[idx])  # Each element is already a [x, y] pair
            
            # Ensure corresponding destination point exists
            if idx * 2 + 1 < len(self.drawing_key_points):
                dst_points.append([self.drawing_key_points[idx * 2], self.drawing_key_points[idx * 2 + 1]])

        # Ensure both lists have the same number of keypoints
        min_points = min(len(src_points), len(dst_points))
        src_points = src_points[:min_points]
        dst_points = dst_points[:min_points]

        # Convert to numpy arrays
        src_points = np.array(src_points, dtype=np.float32).reshape(-1, 2)
        dst_points = np.array(dst_points, dtype=np.float32).reshape(-1, 2)
        # Calculate homography matrix
        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        # Process each frame
        for frame_num, player_bbox in enumerate(player_boxes):
            # Process players
            output_player_bboxes_dict = {}
            
            for player_id, bbox in player_bbox.items():
                # Get foot position of the player
                foot_position = get_foot_position(bbox)
                
                # Convert to homogeneous coordinates
                point = np.array([foot_position[0], foot_position[1], 1], dtype=np.float32).reshape(1, 3)
                
                # Apply homography transformation
                transformed_point = np.dot(H, point.T).T
                
                # Convert back from homogeneous coordinates
                transformed_point = transformed_point / transformed_point[0, 2]
                
                # Save the transformed coordinates
                mini_court_player_position = (transformed_point[0, 0], transformed_point[0, 1])
                output_player_bboxes_dict[player_id] = mini_court_player_position
            
            output_player_boxes.append(output_player_bboxes_dict)
            
            # Process ball
            if frame_num < len(ball_boxes):
                ball_box = ball_boxes[frame_num][1]
                ball_position = get_center_of_bbox(ball_box)
                
                # Convert ball position to homogeneous coordinates
                ball_point = np.array([ball_position[0], ball_position[1], 1], dtype=np.float32).reshape(1, 3)
                
                # Apply homography transformation
                transformed_ball = np.dot(H, ball_point.T).T
                
                # Convert back from homogeneous coordinates
                transformed_ball = transformed_ball / transformed_ball[0, 2]
                
                # Save the transformed coordinates
                mini_court_ball_position = (transformed_ball[0, 0], transformed_ball[0, 1])
                output_ball_boxes.append({1: mini_court_ball_position})
        
        return output_player_boxes, output_ball_boxes
    
    def draw_points_on_mini_court(self,frames,postions, color=(255,255,0)):
        '''Draw points on the mini court'''
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames
    
    
    def draw_player_distance_heatmap(self, frames, player_mini_court_detections, resolution=20, color_map=cv2.COLORMAP_HOT, alpha=0.6):
        """
        Draws a dynamic heatmap on the mini-court showing the distance of each point from the nearest player.
        The heatmap is applied only to the upper part of the court (opponent's half).

        Parameters:
        frames - List of video frames to process
        player_mini_court_detections - Player positions for each frame
        resolution - Grid resolution for the heatmap
        color_map - OpenCV color map to use
        alpha - Heatmap transparency
        """
        # Create grid dimensions
        grid_height = resolution
        grid_width = resolution

        # Calculate step sizes
        x_step = self.court_drawing_width / grid_width
        y_step = (self.court_end_y - self.court_start_y) / grid_height

        # Find the y position of the net (mid-court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)

        # Maximum possible distance for normalization
        max_distance = np.sqrt(self.court_drawing_width**2 + (net_y - self.court_start_y)**2)

        output_frames = []

        # For each frame, create a specific heatmap
        for frame_idx, frame_player_positions in enumerate(player_mini_court_detections):
            if frame_idx >= len(frames):
                break

            # Create the grid for this frame's heatmap
            # Use only the upper half of the grid
            heatmap_grid = np.zeros((grid_height, grid_width))

            # For each grid point, calculate the distance from the nearest player
            for i in range(grid_width):
                for j in range(grid_height):
                    grid_x = self.court_start_x + i * x_step
                    grid_y = self.court_start_y + j * y_step

                    # Apply the heatmap only to the upper part of the court (above the net)
                    if grid_y >= net_y:
                        continue  # Skip pixels in the lower part

                    min_distance = max_distance
                    for player_id, position in frame_player_positions.items():
                        if position is None:
                            continue

                        player_x, player_y = position
                        distance = euclidean_distance((grid_x, grid_y), (player_x, player_y))
                        min_distance = min(min_distance, distance)

                    # Normalize the distance between 0 and 1
                    normalized_distance = min_distance / max_distance
                    heatmap_grid[j, i] = normalized_distance

            # Apply the color map to the grid
            heatmap_colored = cv2.applyColorMap((heatmap_grid * 255).astype(np.uint8), color_map)

            # Resize to court dimensions
            heatmap_resized = cv2.resize(
                heatmap_colored, 
                (self.court_drawing_width, self.court_end_y - self.court_start_y)
            )

            # Create a frame with the heatmap
            frame_copy = frames[frame_idx].copy()

            # Create a transparent overlay of the heatmap
            overlay = frame_copy.copy()

            # Apply the heatmap only to the upper part of the court
            # From the starting position to the net
            overlay[
                self.court_start_y:net_y,  # Only from the upper part to the net
                self.court_start_x:self.court_end_x
            ] = heatmap_resized[0:(net_y-self.court_start_y), :]  # Take only the upper part of the heatmap

            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            output_frames.append(frame_copy)

        # If there are remaining frames, simply copy them
        for i in range(len(output_frames), len(frames)):
            output_frames.append(frames[i].copy())

        return output_frames
    
    
    def draw_ball_landing_points(self, frames, ball_mini_court_detections, landing_frames, color=(0, 255, 255), size=5):
        """
        Draws ball landing points on the mini-court.
        For each segment between consecutive landing frames, shows the future landing point.
        
        Parameters:
        frames - List of video frames to process
        ball_mini_court_detections - Ball positions in mini-court coordinates
        landing_frames - List of frame indices where the ball hits the ground
        color - Color of the landing points (BGR format)
        size - Size of the circle representing the landing point
        """
        output_frames = frames.copy()
        
        # Create intervals based on landing frames
        intervals = []
        for i in range(len(landing_frames)):
            if i == 0:
                start = 0
            else:
                start = landing_frames[i-1] + 1
            
            end = landing_frames[i]
            intervals.append((start, end, landing_frames[i]))
        
        # Add final interval if needed
        if landing_frames[-1] < len(frames) - 1:
            intervals.append((landing_frames[-1] + 1, len(frames) - 1, landing_frames[-1]))
        
        # Process each frame
        for frame_idx in range(len(frames)):
            # Find which interval this frame belongs to
            target_landing_frame = None
            for start, end, landing_frame in intervals:
                if start <= frame_idx <= end:
                    target_landing_frame = landing_frame
                    break
            
            if target_landing_frame is not None and target_landing_frame < len(ball_mini_court_detections):
                # Get the landing position
                if 1 in ball_mini_court_detections[target_landing_frame]:
                    landing_pos = ball_mini_court_detections[target_landing_frame][1]
                    
                    x, y = landing_pos
                    x = int(x)
                    y = int(y)
                    
                    # Draw the landing point
                    cv2.circle(output_frames[frame_idx], (x, y), size, color, -1)
                    
                    # Draw an "X" mark to make it more visible
                    line_length = size - 2
                    cv2.line(output_frames[frame_idx], 
                            (x - line_length, y - line_length),
                            (x + line_length, y + line_length),
                            (0, 0, 255), 2)
                    cv2.line(output_frames[frame_idx],
                            (x - line_length, y + line_length),
                            (x + line_length, y - line_length),
                            (0, 0, 255), 2)
        
        return output_frames

    def draw_shot_trajectories(self, frames, player_mini_court_detections, ball_mini_court_detections, 
                            landing_frames, player_hit_frames,
                            line_color=(0, 255, 255), dot_color=(0, 255, 255), line_thickness=2):
        """
        Draws shot trajectories as dotted lines from lower player hit frames to landing positions.
        
        Parameters:
        frames - List of video frames to process
        player_mini_court_detections - Player positions for each frame
        ball_mini_court_detections - Ball positions for each frame
        landing_frames - List of frame indices where the ball hits the ground
        player_hit_frames - List of frame indices where the lower player hits the ball
        line_color - Color of the trajectory line (BGR format)
        dot_color - Color of the landing point (BGR format)
        line_thickness - Thickness of the trajectory line
        """
        output_frames = frames.copy()
        
        # Find the y position of the net (mid-court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        
        # Map hit frames to nearest landing frames
        hit_to_landing = {}
        for hit_frame in player_hit_frames:
            # Find the next landing after this hit
            next_landing = next((lf for lf in landing_frames if lf > hit_frame), None)
            if next_landing:
                hit_to_landing[hit_frame] = next_landing
        
        # Process each frame
        for frame_idx in range(len(frames)):
            # Find which hit frame this current frame belongs to
            current_hit_frame = None
            current_landing_frame = None
            
            for hit_frame, landing_frame in hit_to_landing.items():
                if hit_frame <= frame_idx < landing_frame:
                    current_hit_frame = hit_frame
                    current_landing_frame = landing_frame
                    break
            
            # If we're in a valid interval between hit and landing
            if current_hit_frame is not None and current_landing_frame is not None:
                # Get the ball position at hit frame
                if current_hit_frame < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[current_hit_frame]:
                    ball_hit_pos = ball_mini_court_detections[current_hit_frame][1]
                    
                    # Get the landing position
                    if current_landing_frame < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[current_landing_frame]:
                        landing_pos = ball_mini_court_detections[current_landing_frame][1]
                        landing_x, landing_y = int(landing_pos[0]), int(landing_pos[1])
                        
                        # Check if the landing position is in the upper part of the court
                        if landing_y < net_y:
                            # Find the closest player to the ball at the hit frame
                            min_distance = float('inf')
                            closest_player_pos = None
                            
                            for player_id, position in player_mini_court_detections[current_hit_frame].items():
                                if position is not None:
                                    dist = euclidean_distance(position, ball_hit_pos)
                                    if dist < min_distance:
                                        min_distance = dist
                                        closest_player_pos = position
                            
                            if closest_player_pos is not None:
                                player_x, player_y = int(closest_player_pos[0]), int(closest_player_pos[1])
                                
                                # Check if player is in the lower part of the court
                                if player_y >= net_y:
                                    # Draw the landing point
                                    cv2.circle(output_frames[frame_idx], (landing_x, landing_y), 5, dot_color, -1)
                                    
                                    # Draw dotted line from player to landing position
                                    line_length = np.sqrt((landing_x - player_x)**2 + (landing_y - player_y)**2)
                                    num_segments = int(line_length / 10)  # One segment every ~10 pixels
                                    
                                    if num_segments > 1:
                                        # Create points along the line
                                        for i in range(num_segments):
                                            t = i / (num_segments - 1)
                                            dot_x = int(player_x + t * (landing_x - player_x))
                                            dot_y = int(player_y + t * (landing_y - player_y))
                                            
                                            # Draw a small line segment (dot) at this position
                                            if i % 2 == 0:  # Skip every other segment for dotted effect
                                                cv2.circle(output_frames[frame_idx], (dot_x, dot_y), 1, line_color, line_thickness)
        
        return output_frames

    def draw_ball_landing_heatmap(self, frames, player_mini_court_detections, ball_mini_court_detections, 
                                landing_frames, player_hit_frames,
                                sigma=15, alpha=0.7, color_map=cv2.COLORMAP_HOT, fade_frames=60):
        """
        Creates a dynamic Gaussian heatmap showing ball landing areas from lower player hits.
        For each shot from the lower player, shows a NEW heatmap for the next landing point.
        Hides the heatmap completely when a player in the upper part of the court hits the ball.
        
        Parameters:
        frames - List of video frames to process
        player_mini_court_detections - Player positions for each frame
        ball_mini_court_detections - Ball positions in mini-court coordinates
        landing_frames - List of frame indices where the ball hits the ground
        player_hit_frames - List of frame indices where the lower player hits the ball
        sigma - Standard deviation for Gaussian blur (controls the spread of each landing point)
        alpha - Transparency of the heatmap
        color_map - OpenCV colormap to use for visualization
        fade_frames - Number of frames over which a landing point gradually appears
        """
        output_frames = frames.copy()
        
        # Court dimensions
        court_width = self.court_end_x - self.court_start_x
        court_height = self.court_end_y - self.court_start_y
        
        # Find the y position of the net (mid-court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        
        # Determine the sequence of hits (who hit when) and their landing points
        hit_sequence = []
        
        # Process all landing frames to detect hits
        for i, landing_frame in enumerate(landing_frames):
            if i == 0:
                continue  # Skip first landing as we don't know who hit before it
                    
            # Find the previous landing
            prev_landing = landing_frames[i-1]
            
            # Determine mid-point or most likely frame where ball was hit after previous landing
            hit_frame = prev_landing + max(1, (landing_frame - prev_landing) // 3)
            
            # Find who hit the ball (closest player to ball)
            if hit_frame < len(player_mini_court_detections) and hit_frame < len(ball_mini_court_detections):
                if 1 in ball_mini_court_detections[hit_frame]:
                    ball_pos = ball_mini_court_detections[hit_frame][1]
                    
                    # Find closest player and check if they're above or below net
                    min_dist = float('inf')
                    closest_player_y = None
                    
                    for player_id, position in player_mini_court_detections[hit_frame].items():
                        if position is not None:
                            dist = euclidean_distance(position, ball_pos)
                            if dist < min_dist:
                                min_dist = dist
                                closest_player_y = position[1]
                    
                    if closest_player_y is not None:
                        # Add to hit sequence: hit_frame, is_upper_player, landing_frame
                        is_upper_player = closest_player_y < net_y
                        
                        # Get the landing position
                        landing_pos = None
                        if landing_frame < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[landing_frame]:
                            landing_pos = ball_mini_court_detections[landing_frame][1]
                            landing_x = int(landing_pos[0]) - self.court_start_x
                            landing_y = int(landing_pos[1]) - self.court_start_y
                            
                            # Only consider valid landings
                            if 0 <= landing_x < court_width and 0 <= landing_y < court_height:
                                hit_sequence.append((hit_frame, is_upper_player, landing_frame, (landing_x, landing_y)))
        
        # Sort hits by frame number
        hit_sequence.sort(key=lambda x: x[0])
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Find the active hit (the most recent hit before or at this frame)
            active_hit = None
            next_hit = None
            
            for i, (hit_frame, is_upper, landing_frame, landing_pos) in enumerate(hit_sequence):
                if hit_frame <= frame_idx:
                    active_hit = (hit_frame, is_upper, landing_frame, landing_pos)
                    
                    # Find the next hit
                    if i + 1 < len(hit_sequence):
                        next_hit = hit_sequence[i + 1]
                else:
                    break
            
            # No active hit yet or active hit is from upper player
            if active_hit is None or active_hit[1]:  # is_upper
                output_frames[frame_idx] = frame.copy()
                continue
            
            # We have an active hit from lower player - show heatmap for its landing point
            hit_frame, is_upper, landing_frame, landing_pos = active_hit
            landing_x, landing_y = landing_pos
            
            # Only show heatmap for landings in upper part of court
            if landing_y >= (net_y - self.court_start_y):
                output_frames[frame_idx] = frame.copy()
                continue
            
            # Calculate frame range where this heatmap should be visible
            # From current hit to next hit (or end of frames)
            end_frame = len(frames)
            if next_hit:
                end_frame = min(end_frame, next_hit[0])
            
            # Create heatmap for this single landing point with fade-in effect
            frames_since_hit = frame_idx - hit_frame
            
            # Apply fade-in effect
            if frames_since_hit < fade_frames:
                intensity = min(1.0, frames_since_hit / fade_frames)
            else:
                intensity = 1.0
            
            heatmap = np.zeros((court_height, court_width), dtype=np.float32)
            
            # Add the single point to heatmap
            point_map = np.zeros((court_height, court_width), dtype=np.float32)
            point_map[landing_y, landing_x] = intensity
            
            # Apply Gaussian blur
            point_map = cv2.GaussianBlur(point_map, (0, 0), sigma)
            
            # Add to heatmap
            heatmap += point_map
            
            # Finalize and display heatmap
            if np.max(heatmap) > 0:
                # Normalize heatmap to range 0-1
                heatmap = heatmap / np.max(heatmap)
                
                # Apply colormap
                heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), color_map)
                
                # Create mask for overlay
                result = frame.copy()
                mask = np.zeros_like(result)
                
                # Apply heatmap only to upper court
                mask[
                    self.court_start_y:net_y,
                    self.court_start_x:self.court_end_x
                ] = heatmap_colored[0:(net_y-self.court_start_y), :]
                
                # Blend with original frame
                cv2.addWeighted(mask, alpha, result, 1, 0, result)
                output_frames[frame_idx] = result
            else:
                output_frames[frame_idx] = frame.copy()
        
        return output_frames
    

    def create_player_heatmap_animation(self, player_mini_court_detections, ball_mini_court_detections=None, 
                                    output_path="output/animations/player_heatmap_animation.mp4", 
                                    player_sigma=10, color_map=cv2.COLORMAP_HOT, alpha=0.7, fps=15,
                                    draw_players=True, player_reach_threshold=0.25, easy_reach_threshold=0.1, save_mask=False, 
                                    mask_path="output/masks/player_heatmap_mask.npy"):
        """
        Creates an animation showing only the player distance heatmap on the mini-court.

        Parameters:
        player_mini_court_detections - Player positions for each frame
        ball_mini_court_detections - Ignored, present only for compatibility with other functions
        output_path - Path to save the animation
        player_sigma - Standard deviation for Gaussian blur of the heatmap
        color_map - OpenCV color map to use for visualization
        alpha - Heatmap transparency
        fps - Frames per second of the animation
        draw_players - Whether to draw player positions
        player_reach_threshold - Threshold to determine the "easy reach" distance for players
        easy_reach_threshold - Threshold to identify easily reachable areas
        save_mask - Whether to save raw heatmap data
        mask_path - Path to save the heatmap data

        Returns:
        output_path - Path of the created animation
        mask_data - Raw heatmap data if save_mask is True
        """
        # Create the video writer
        width = self.drawing_rectangle_width
        height = self.drawing_rectangle_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Find the y position of the net (mid-court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        
        # Court dimensions
        court_width = self.court_end_x - self.court_start_x
        court_height = self.court_end_y - self.court_start_y
        
        # Upper court dimensions
        upper_court_height = net_y - self.court_start_y
        upper_court_width = self.court_drawing_width
        
        # Space for heatmap data if requested
        if save_mask:
            mask_data = []
        
        # Maximum possible distance for normalization
        max_distance = np.sqrt(self.court_drawing_width**2 + (net_y - self.court_start_y)**2)
        
        # For each frame, create the heatmap
        frame_count = len(player_mini_court_detections)
        for frame_idx in range(frame_count):
            # Create an empty image for the court
            court_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw the court lines (white)
            for line in self.lines:
                start_point = (int(self.drawing_key_points[line[0]*2]) - self.start_x, 
                            int(self.drawing_key_points[line[0]*2+1]) - self.start_y)
                end_point = (int(self.drawing_key_points[line[1]*2]) - self.start_x, 
                        int(self.drawing_key_points[line[1]*2+1]) - self.start_y)
                cv2.line(court_image, start_point, end_point, (255, 255, 255), 2)

            # Draw the net (purple)
            net_start_point = (self.drawing_key_points[0] - self.start_x, 
                            int((self.drawing_key_points[1] + self.drawing_key_points[5])/2) - self.start_y)
            net_end_point = (self.drawing_key_points[2] - self.start_x, 
                        int((self.drawing_key_points[1] + self.drawing_key_points[5])/2) - self.start_y)
            cv2.line(court_image, net_start_point, net_end_point, (255, 0, 255), 2)
            
            # ----- PLAYER DISTANCE HEATMAP -----
            # Create the grid for player distance
            player_heatmap = np.zeros((upper_court_height, upper_court_width), dtype=np.float32)
            
            if frame_idx < len(player_mini_court_detections):
                # Calculate the distance from players for each point in the upper court
                for y in range(upper_court_height):
                    for x in range(upper_court_width):
                        grid_y = self.court_start_y + y
                        grid_x = self.court_start_x + x
                        
                        # Measure the distance from this point to the nearest player
                        min_distance = max_distance
                        for player_id, position in player_mini_court_detections[frame_idx].items():
                            if position is not None:
                                player_x, player_y = position
                                distance = euclidean_distance((grid_x, grid_y), (player_x, player_y))
                                min_distance = min(min_distance, distance)
                        
                        # Normalize the distance within [0, 1]
                        normalized_distance = (min_distance / max_distance)**4
                        
                        # Apply threshold to significantly reduce values near players
                        if (min_distance / max_distance) < player_reach_threshold:
                            # Apply a sharp drop for points within player reach
                            normalized_distance *= 0.1
                        
                        player_heatmap[y, x] = normalized_distance
                
                # Apply Gaussian blur to smooth the heatmap
                if player_sigma > 0:
                    player_heatmap = cv2.GaussianBlur(player_heatmap, (0, 0), player_sigma)
                    
                # Renormalize to ensure the maximum value is 1
                if np.max(player_heatmap) > 0:
                    player_heatmap = player_heatmap / np.max(player_heatmap)
            
            # Save raw heatmap data if requested
            if save_mask:
                mask_data.append(player_heatmap.copy())
            
            # Apply color map to the heatmap
            colored_heatmap = cv2.applyColorMap((player_heatmap * 255).astype(np.uint8), color_map)
            
            # Create a mask for overlay
            mask = np.zeros_like(court_image)
            
            # Apply the heatmap to the upper part of the court
            mask[
                self.court_start_y - self.start_y:net_y - self.start_y,
                self.court_start_x - self.start_x:self.court_end_x - self.start_x
            ] = colored_heatmap
            
            # Blend with the court image
            cv2.addWeighted(mask, alpha, court_image, 1, 0, court_image)
            
            # Draw players as points
            if draw_players:
                for player_id, position in player_mini_court_detections[frame_idx].items():
                    if position is not None:
                        x, y = position
                        x_adj = int(x - self.start_x)
                        y_adj = int(y - self.start_y)
                        if 0 <= x_adj < width and 0 <= y_adj < height:
                            cv2.circle(court_image, (x_adj, y_adj), 5, (255, 255, 0), -1)  # Yellow for players
            
            # Write the frame to the video
            video_writer.write(court_image)
        
        # Release the writer
        video_writer.release()
        
        # Save the mask data if requested
        if save_mask:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, np.array(mask_data))
            return output_path, mask_data
        
        return output_path


    def create_ball_heatmap_animation(self, player_mini_court_detections, ball_mini_court_detections, landing_frames, 
                                    player_hit_frames, output_path="output/animations/ball_heatmap_animation.mp4",
                                    sigma=8, color_map=cv2.COLORMAP_HOT, alpha=0.7, fps=15, draw_ball=False,
                                    save_mask=False, mask_path="output/masks/ball_heatmap_mask.npy"):
        """
        Creates an animation showing the mini-court with players, shot trajectories, and ball landing heatmap.
        
        Parameters:
        player_mini_court_detections - Player positions for each frame
        ball_mini_court_detections - Ball positions for each frame
        landing_frames - List of frame indices where the ball hits the ground
        player_hit_frames - List of frame indices where the lower player hits the ball
        output_path - Path where to save the animation
        sigma - Standard deviation for Gaussian blur (controls the spread of each landing point)
        color_map - OpenCV color map to use for visualization
        alpha - Heatmap transparency
        fps - Frames per second of the animation
        draw_ball - Whether to draw the ball position
        save_mask - Whether to save the raw heatmap data for combining later
        mask_path - Path where to save the heatmap mask data
        
        Returns:
        output_path - Path to the created animation
        mask_data - Raw heatmap data if save_mask is True
        """
        # Create a video writer
        width = self.drawing_rectangle_width
        height = self.drawing_rectangle_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Find the y position of the net (mid-court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        
        # Court dimensions
        court_width = self.court_end_x - self.court_start_x
        court_height = self.court_end_y - self.court_start_y
        
        # Upper court dimensions (used for mask storage)
        upper_court_height = net_y - self.court_start_y
        upper_court_width = self.court_drawing_width
        
        # Storage for heatmap masks if needed
        if save_mask:
            mask_data = []
        
        # Determine the sequence of hits (who hit when) and their landing points
        hit_sequence = []
        
        # Process all landing frames to detect hits
        for i, landing_frame in enumerate(landing_frames):
            if i == 0:
                continue  # Skip first landing as we don't know who hit before it
                    
            # Find the previous landing
            prev_landing = landing_frames[i-1]
            
            # Determine mid-point or most likely frame where ball was hit after previous landing
            hit_frame = prev_landing + max(1, (landing_frame - prev_landing) // 3)
            
            # Find who hit the ball (closest player to ball)
            if hit_frame < len(player_mini_court_detections) and hit_frame < len(ball_mini_court_detections):
                if 1 in ball_mini_court_detections[hit_frame]:
                    ball_pos = ball_mini_court_detections[hit_frame][1]
                    
                    # Find closest player and check if they're above or below net
                    min_dist = float('inf')
                    closest_player_pos = None
                    closest_player_y = None
                    
                    for player_id, position in player_mini_court_detections[hit_frame].items():
                        if position is not None:
                            dist = euclidean_distance(position, ball_pos)
                            if dist < min_dist:
                                min_dist = dist
                                closest_player_pos = position
                                closest_player_y = position[1]
                    
                    if closest_player_y is not None:
                        # Add to hit sequence: hit_frame, is_upper_player, landing_frame, player_pos, landing_pos
                        is_upper_player = closest_player_y < net_y
                        
                        # Get the landing position
                        landing_pos = None
                        if landing_frame < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[landing_frame]:
                            landing_pos = ball_mini_court_detections[landing_frame][1]
                            landing_x = int(landing_pos[0]) - self.court_start_x
                            landing_y = int(landing_pos[1]) - self.court_start_y
                            
                            # Only consider valid landings
                            if 0 <= landing_x < court_width and 0 <= landing_y < court_height:
                                hit_sequence.append((hit_frame, is_upper_player, landing_frame, closest_player_pos, (landing_x, landing_y)))
        
        # Sort hits by frame number
        hit_sequence.sort(key=lambda x: x[0])
        
        # For each frame in the animation sequence
        frame_count = len(ball_mini_court_detections)
        for frame_idx in range(frame_count):
            # Create a blank black image for the court background
            court_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a blank heatmap for this frame
            upper_court_heatmap = np.zeros((upper_court_height, upper_court_width), dtype=np.float32)
            
            # Draw court lines (white)
            for line in self.lines:
                start_point = (int(self.drawing_key_points[line[0]*2]) - self.start_x, 
                            int(self.drawing_key_points[line[0]*2+1]) - self.start_y)
                end_point = (int(self.drawing_key_points[line[1]*2]) - self.start_x, 
                        int(self.drawing_key_points[line[1]*2+1]) - self.start_y)
                cv2.line(court_image, start_point, end_point, (255, 255, 255), 2)

            # Draw the net (purple)
            net_start_point = (self.drawing_key_points[0] - self.start_x, 
                            int((self.drawing_key_points[1] + self.drawing_key_points[5])/2) - self.start_y)
            net_end_point = (self.drawing_key_points[2] - self.start_x, 
                        int((self.drawing_key_points[1] + self.drawing_key_points[5])/2) - self.start_y)
            cv2.line(court_image, net_start_point, net_end_point, (255, 0, 255), 2)
            
            # Find the active hit (the most recent hit before or at this frame)
            active_hit = None
            next_hit = None
            
            for i, (hit_frame, is_upper, landing_frame, player_pos, landing_pos) in enumerate(hit_sequence):
                if hit_frame <= frame_idx:
                    active_hit = (hit_frame, is_upper, landing_frame, player_pos, landing_pos)
                    
                    # Find the next hit
                    if i + 1 < len(hit_sequence):
                        next_hit = hit_sequence[i + 1]
                else:
                    break
            
            # Apply ball_landing_heatmap logic
            if active_hit is not None and not active_hit[1]:  # not is_upper (from lower player)
                hit_frame, is_upper, landing_frame, player_pos, landing_pos = active_hit
                landing_x, landing_y = landing_pos
                
                # Only show heatmap for landings in upper part of court
                if landing_y < (net_y - self.court_start_y):
                    # Create heatmap for this single landing point
                    frames_since_hit = frame_idx - hit_frame
                    
                    # Apply fade-in effect (only for 60 frames)
                    fade_frames = 60
                    if frames_since_hit < fade_frames:
                        intensity = min(1.0, frames_since_hit / fade_frames)
                    else:
                        intensity = 1.0
                    
                    heatmap = np.zeros((court_height, court_width), dtype=np.float32)
                    
                    # Add the single point to heatmap
                    point_map = np.zeros((court_height, court_width), dtype=np.float32)
                    point_map[landing_y, landing_x] = intensity
                    
                    # Apply Gaussian blur
                    point_map = cv2.GaussianBlur(point_map, (0, 0), sigma)
                    
                    # Add to heatmap
                    heatmap += point_map
                    
                    # Store the upper part for mask data
                    upper_court_heatmap = heatmap[0:upper_court_height, 0:upper_court_width].copy()
                    
                    # Finalize and display heatmap
                    if np.max(heatmap) > 0:
                        # Normalize heatmap to range 0-1
                        heatmap = heatmap / np.max(heatmap)
                        
                        # Apply colormap
                        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), color_map)
                        
                        # Create mask for overlay
                        mask = np.zeros_like(court_image)
                        
                        # Apply heatmap only to upper court
                        mask[
                            self.court_start_y - self.start_y:net_y - self.start_y,
                            self.court_start_x - self.start_x:self.court_end_x - self.start_x
                        ] = heatmap_colored[0:upper_court_height, :]
                        
                        # Blend with original frame
                        cv2.addWeighted(mask, alpha, court_image, 1, 0, court_image)
            
            # Save the raw upper court heatmap data if requested
            if save_mask:
                mask_data.append(upper_court_heatmap)
                
            # Apply draw_shot_trajectories logic
            for hit_frame, is_upper, landing_frame, player_pos, landing_pos in hit_sequence:
                # Only show trajectories from lower player hits
                if not is_upper and hit_frame <= frame_idx <= landing_frame:
                    # Landing position is already in relative coordinates to court_start
                    landing_x, landing_y = landing_pos
                    landing_x_abs = int(landing_x + self.court_start_x - self.start_x)
                    landing_y_abs = int(landing_y + self.court_start_y - self.start_y)
                    
                    # Check if the landing position is in the upper part of the court
                    if landing_y < (net_y - self.court_start_y):
                        # Player position is in absolute coordinates
                        player_x, player_y = int(player_pos[0]) - self.start_x, int(player_pos[1]) - self.start_y
                        
                        # Draw the landing point
                        cv2.circle(court_image, (landing_x_abs, landing_y_abs), 5, (0, 255, 255), -1)
                        
                        # Draw dotted line from player to landing position
                        line_length = np.sqrt((landing_x_abs - player_x)**2 + (landing_y_abs - player_y)**2)
                        num_segments = int(line_length / 10)  # One segment every ~10 pixels
                        
                        if num_segments > 1:
                            # Create points along the line
                            for i in range(num_segments):
                                t = i / (num_segments - 1)
                                dot_x = int(player_x + t * (landing_x_abs - player_x))
                                dot_y = int(player_y + t * (landing_y_abs - player_y))
                                
                                # Draw a small line segment (dot) at this position
                                if i % 2 == 0:  # Skip every other segment for dotted effect
                                    cv2.circle(court_image, (dot_x, dot_y), 1, (0, 255, 255), 2)
            
            # Draw ball current position
            if draw_ball:            
                if frame_idx < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[frame_idx]:
                    ball_pos = ball_mini_court_detections[frame_idx][1]
                    ball_x, ball_y = int(ball_pos[0]) - self.start_x, int(ball_pos[1]) - self.start_y
                    if 0 <= ball_x < width and 0 <= ball_y < height:
                        cv2.circle(court_image, (ball_x, ball_y), 4, (0, 255, 255), -1)
            
            # Draw players
            if frame_idx < len(player_mini_court_detections):
                for player_id, position in player_mini_court_detections[frame_idx].items():
                    if position is not None:
                        x, y = int(position[0]) - self.start_x, int(position[1]) - self.start_y
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(court_image, (x, y), 5, (255, 255, 0), -1)
            
            # Write frame to video
            video_writer.write(court_image)
        
        # Release the writer
        video_writer.release()
        
        # Save mask data if requested
        if save_mask:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, np.array(mask_data))
            return output_path, mask_data
        
        return output_path
    
    def create_scoring_heatmap_animation(self, player_mini_court_detections, ball_mini_court_detections, landing_frames,
                                        output_path="output/animations/scoring_heatmap_animation.mp4",
                                        player_sigma=10, ball_sigma=8, color_map=cv2.COLORMAP_JET, alpha=0.7, fps=15,
                                        draw_trajectory=True, trajectory_color=(0, 255, 255), line_thickness=2,
                                        easy_reach_threshold = 0.1, save_mask=False,
                                        mask_path="output/masks/scoring_heatmap_mask.npy"):
        """
        Creates an animation showing the scoring probability heatmap - combining player distance and ball landing probability.
        When the ball is hit by a player in the upper court, only the player distance heatmap is shown.
        
        Parameters:
        player_mini_court_detections - Player positions for each frame
        ball_mini_court_detections - Ball positions for each frame
        landing_frames - List of frame indices where the ball hits the ground
        output_path - Path where to save the animation
        player_sigma - Standard deviation for player distance Gaussian blur
        ball_sigma - Standard deviation for ball landing Gaussian blur
        color_map - OpenCV color map to use for visualization
        alpha - Heatmap transparency
        fps - Frames per second of the animation
        draw_trajectory - Whether to draw shot trajectories
        trajectory_color - Color of the trajectory line (BGR format)
        line_thickness - Thickness of the trajectory line
        save_mask - Whether to save the raw heatmap data
        mask_path - Path where to save the heatmap mask data
        
        Returns:
        output_path - Path to the created animation
        mask_data - Raw heatmap data if save_mask is True
        """
        # Create a video writer
        width = self.drawing_rectangle_width
        height = self.drawing_rectangle_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Find the y position of the net (mid-court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        
        # Court dimensions
        court_width = self.court_end_x - self.court_start_x
        court_height = self.court_end_y - self.court_start_y
        
        # Upper court dimensions
        upper_court_height = net_y - self.court_start_y
        upper_court_width = self.court_drawing_width
        
        # Storage for combined heatmap masks if needed
        if save_mask:
            mask_data = []
        
        # Maximum possible distance for normalization
        max_distance = np.sqrt(self.court_drawing_width**2 + (net_y - self.court_start_y)**2)
        
        # Determine the sequence of hits and landings
        hit_sequence = []
        for i, landing_frame in enumerate(landing_frames):
            if i == 0:
                continue  # Skip first landing
                    
            # Find the previous landing
            prev_landing = landing_frames[i-1]
            
            # Estimate when the ball was hit after previous landing
            hit_frame = prev_landing + max(1, (landing_frame - prev_landing) // 3)
            
            # Find who hit the ball
            if hit_frame < len(player_mini_court_detections) and hit_frame < len(ball_mini_court_detections):
                if 1 in ball_mini_court_detections[hit_frame]:
                    ball_pos = ball_mini_court_detections[hit_frame][1]
                    
                    # Find closest player
                    min_dist = float('inf')
                    closest_player_y = None
                    closest_player_pos = None
                    
                    for player_id, position in player_mini_court_detections[hit_frame].items():
                        if position is not None:
                            dist = euclidean_distance(position, ball_pos)
                            if dist < min_dist:
                                min_dist = dist
                                closest_player_y = position[1]
                                closest_player_pos = position
                    
                    if closest_player_y is not None:
                        # Add to sequence
                        is_upper_player = closest_player_y < net_y
                        
                        # Get the landing position
                        if landing_frame < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[landing_frame]:
                            landing_pos = ball_mini_court_detections[landing_frame][1]
                            landing_x = int(landing_pos[0]) - self.court_start_x
                            landing_y = int(landing_pos[1]) - self.court_start_y
                            
                            # Only consider valid landings
                            if 0 <= landing_x < court_width and 0 <= landing_y < court_height:
                                hit_sequence.append((hit_frame, is_upper_player, landing_frame, closest_player_pos, (landing_x, landing_y)))
        
        # Sort hits by frame number
        hit_sequence.sort(key=lambda x: x[0])
        
        # For each frame, create heatmap
        frame_count = len(player_mini_court_detections)
        for frame_idx in range(frame_count):
            # Create blank image for court
            court_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw court lines (white)
            for line in self.lines:
                start_point = (int(self.drawing_key_points[line[0]*2]) - self.start_x, 
                            int(self.drawing_key_points[line[0]*2+1]) - self.start_y)
                end_point = (int(self.drawing_key_points[line[1]*2]) - self.start_x, 
                        int(self.drawing_key_points[line[1]*2+1]) - self.start_y)
                cv2.line(court_image, start_point, end_point, (255, 255, 255), 2)

            # Draw the net (purple)
            net_start_point = (self.drawing_key_points[0] - self.start_x, 
                            int((self.drawing_key_points[1] + self.drawing_key_points[5])/2) - self.start_y)
            net_end_point = (self.drawing_key_points[2] - self.start_x, 
                        int((self.drawing_key_points[1] + self.drawing_key_points[5])/2) - self.start_y)
            cv2.line(court_image, net_start_point, net_end_point, (255, 0, 255), 2)
            
            # ----- PLAYER DISTANCE HEATMAP -----
            # Create grid for player distance
            player_heatmap = np.zeros((upper_court_height, upper_court_width), dtype=np.float32)
            
            if frame_idx < len(player_mini_court_detections):
                # Calculate player distance for each point in the upper court
                for y in range(upper_court_height):
                    for x in range(upper_court_width):
                        grid_y = self.court_start_y + y
                        grid_x = self.court_start_x + x
                        
                        # Measure distance from this point to nearest player
                        min_distance = max_distance
                        for player_id, position in player_mini_court_detections[frame_idx].items():
                            if position is not None:
                                player_x, player_y = position
                                distance = euclidean_distance((grid_x, grid_y), (player_x, player_y))
                                min_distance = min(min_distance, distance)
                        
                        # Normalize distance to [0, 1]
                        normalized_distance = (min_distance / max_distance)**4
                        
                        # Apply a threshold to significantly reduce values near players
                        # Distance threshold (in normalized units) below which values are greatly reduced
                        player_reach_threshold = 0.25
                        if (min_distance / max_distance) < player_reach_threshold:
                            # Apply steep drop-off for points within player reach
                            normalized_distance *= 0.1
                        
                        player_heatmap[y, x] = normalized_distance
                
                # Optional: Apply Gaussian blur to smooth player heatmap
                if player_sigma > 0:
                    player_heatmap = cv2.GaussianBlur(player_heatmap, (0, 0), player_sigma)
                    
                # Re-normalize to ensure max value is 1
                if np.max(player_heatmap) > 0:
                    player_heatmap = player_heatmap / np.max(player_heatmap)
            
            # Find the active hit (most recent hit before current frame)
            active_hit = None
            for hit_frame, is_upper, landing_frame, player_pos, landing_pos in hit_sequence:
                if hit_frame <= frame_idx:
                    active_hit = (hit_frame, is_upper, landing_frame, player_pos, landing_pos)
                else:
                    break
            
            # ----- BALL LANDING HEATMAP (only if the last shot is from the upper court) -----
            combined_heatmap = player_heatmap.copy()  # Default to player heatmap only
            
            # If we have an active hit and it's NOT from upper player, create combined heatmap
            if active_hit is not None and not active_hit[1]:  # not is_upper
                # Initialize ball landing heatmap
                ball_heatmap = np.zeros((upper_court_height, upper_court_width), dtype=np.float32)
                
                hit_frame, is_upper, landing_frame, player_pos, landing_pos = active_hit
                landing_x, landing_y = landing_pos
                
                # Only process landings in upper court
                if landing_y < upper_court_height:
                    # Apply fade-in effect
                    frames_since_hit = frame_idx - hit_frame
                    fade_frames = 60  # Adjust as needed
                    intensity = min(1.0, frames_since_hit / fade_frames) if frames_since_hit < fade_frames else 1.0
                    
                    # Add the landing point
                    point_map = np.zeros((upper_court_height, upper_court_width), dtype=np.float32)
                    if 0 <= landing_y < point_map.shape[0] and 0 <= landing_x < point_map.shape[1]:
                        point_map[landing_y, landing_x] = intensity
                    
                    # Apply Gaussian blur to spread the landing probability
                    if ball_sigma > 0:
                        ball_heatmap = cv2.GaussianBlur(point_map, (0, 0), ball_sigma)
                        
                        # Normalize to [0, 1]
                        if np.max(ball_heatmap) > 0:
                            ball_heatmap = ball_heatmap / np.max(ball_heatmap)
        
                    
                    # Create mask to identify areas easily reachable by players
                    easy_reach_mask = (player_heatmap < easy_reach_threshold)
                    
                    # First, start by combining player and ball heatmaps
                    # Multiply them to get high values only where BOTH:
                    # - Players are far away (high player_heatmap)
                    # - Ball is likely to land (high ball_heatmap)
                    combined_heatmap = player_heatmap * ball_heatmap
                    
                    # Completely zero out values in areas easily reachable by players
                    # regardless of ball landing probability
                    combined_heatmap[easy_reach_mask] = 0.0
                    
                    # Apply additional weighting to emphasize distance from players
                    # This ensures areas far from players get emphasized
                    combined_heatmap = combined_heatmap * (player_heatmap ** 2)
                    
                    # Normalize the final heatmap
                    if np.max(combined_heatmap) > 0:
                        combined_heatmap = combined_heatmap / np.max(combined_heatmap)
                    else:
                        # If everything got zeroed out, just use a very low constant value
                        combined_heatmap = np.ones_like(combined_heatmap) * 0.01
            
            # Save the raw heatmap if requested
            if save_mask:
                mask_data.append(combined_heatmap.copy())
            
            # Apply colormap to the selected heatmap
            colored_heatmap = cv2.applyColorMap((combined_heatmap * 255).astype(np.uint8), color_map)
            
            # Create mask for overlay
            mask = np.zeros_like(court_image)
            
            # Apply heatmap to upper court
            mask[
                self.court_start_y - self.start_y:net_y - self.start_y,
                self.court_start_x - self.start_x:self.court_end_x - self.start_x
            ] = colored_heatmap
            
            # Blend with court image
            cv2.addWeighted(mask, alpha, court_image, 1, 0, court_image)
            
            # Draw trajectory if requested
            if draw_trajectory:
                for hit_frame, is_upper, landing_frame, player_pos, landing_coords in hit_sequence:
                    # Only show trajectories from lower player hits
                    if not is_upper and hit_frame <= frame_idx <= landing_frame:
                        # Get original player position
                        player_x, player_y = int(player_pos[0]) - self.start_x, int(player_pos[1]) - self.start_y
                        
                        # Get landing position
                        landing_x, landing_y = landing_coords
                        landing_x_abs = int(landing_x + self.court_start_x - self.start_x)
                        landing_y_abs = int(landing_y + self.court_start_y - self.start_y)
                        
                        # Check if landing position is in the upper part of the court
                        if landing_y < (net_y - self.court_start_y):
                            # Draw the landing point
                            cv2.circle(court_image, (landing_x_abs, landing_y_abs), 5, trajectory_color, -1)
                            
                            # Draw dotted line from player to landing position
                            line_length = np.sqrt((landing_x_abs - player_x)**2 + (landing_y_abs - player_y)**2)
                            num_segments = int(line_length / 10)  # One segment every ~10 pixels
                            
                            if num_segments > 1:
                                # Create points along the line
                                for i in range(num_segments):
                                    t = i / (num_segments - 1)
                                    dot_x = int(player_x + t * (landing_x_abs - player_x))
                                    dot_y = int(player_y + t * (landing_y_abs - player_y))
                                    
                                    # Draw a small line segment (dot) at this position
                                    if i % 2 == 0:  # Skip every other segment for dotted effect
                                        cv2.circle(court_image, (dot_x, dot_y), 1, trajectory_color, line_thickness)
            
            # Draw players as points
            for player_id, position in player_mini_court_detections[frame_idx].items():
                if position is not None:
                    x, y = position
                    x_adj = int(x - self.start_x)
                    y_adj = int(y - self.start_y)
                    if 0 <= x_adj < width and 0 <= y_adj < height:
                        cv2.circle(court_image, (x_adj, y_adj), 5, (255, 255, 0), -1)  # Yellow for players
            
            
            # Write frame to video
            video_writer.write(court_image)
        
        # Release the writer
        video_writer.release()
        
        # Save mask data if requested
        if save_mask:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, np.array(mask_data))
            return output_path, mask_data
        
        return output_path