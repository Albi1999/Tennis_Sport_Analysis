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
    get_center_of_bbox,
    convert_to_heatmap_values,
    convert_to_pixel_values,
    apply_colormap,
    compute_score_heatmap,
    compute_score_probability,
    test_img_values,
    test_heatmap_values
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250          # Width of the mini court in pixels (depends on the image size)
        self.drawing_rectangle_height = 550         # Height of the mini court in pixels (depends on the image size)
        self.buffer = 70                            # Distance from the right and top of the frame to the mini court
        self.padding_court= 30                      # Padding from the mini court to the court (depends on the image size)

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

        # Compute the height of the court in pixels
        court_height_pixels = self.convert_meters_to_pixels(info.HALF_COURT_LINE_HEIGHT*2)
        
        # Compute the vertical padding to center the court in the mini court area
        vertical_padding = (self.court_end_y - self.court_start_y - court_height_pixels) / 2
        
        # Court Key Points
        # point 0 --> top left corner
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y + vertical_padding)
        # point 1 --> top right corner
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y + vertical_padding)
        # point 2 --> bottom left corner
        drawing_key_points[4] = int(self.court_start_x) 
        drawing_key_points[5] = int(self.court_start_y + vertical_padding + court_height_pixels)
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
        self.net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5])/2)


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

        # Net (Blue)
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

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
    
    def draw_points_on_mini_court(self, frames, positions, object_type='ball'):
        """
        Draw points on the mini court with different colors for upper and lower players.
        
        Args:
            frames (list): List of frames to draw the points on
            positions (list): List of dictionaries containing player positions in mini court
            color (tuple): Color for the points (BGR format) - used for non-player objects
            object_type (str): Type of object ('player' or 'ball')
        
        Returns:
            list: List of frames with points applied
        """
        output_frames = frames.copy()
        
        # Define colors for players
        upper_player_color = (255, 255, 0)  # Cyan in BGR
        lower_player_color = (255, 0, 255)  # Magenta in BGR
        
        # Get the net y-coordinate (dividing line between upper and lower court)
        net_y = self.net_y
        
        if object_type == 'player':
            for frame_num, frame in enumerate(output_frames):
                for player_id, position in positions[frame_num].items():
                    x, y = position
                    x = int(x)
                    y = int(y)
                    
                    # Determine player color based on position
                    if y < net_y:
                        # Player is in upper half of court
                        point_color = upper_player_color
                    else:
                        # Player is in lower half of court
                        point_color = lower_player_color
                        
                    cv2.circle(frame, (x, y), 5, point_color, -1)
        elif object_type == 'ball':
            # For ball, use the specified color
            for frame_num, frame in enumerate(output_frames):
                for _, position in positions[frame_num].items():
                    x, y = position
                    x = int(x)
                    y = int(y)
                    cv2.circle(frame, (x, y), 5, (0,255,255), -1)
        else:
            raise ValueError("Invalid object type. Use 'player' or 'ball'.")
        
        return output_frames
    
    def draw_shot_trajectories(self, frames, player_mini_court_detections, ball_mini_court_detections,
                            ball_landing_frames, ball_shots_frames, serve_player,
                            line_color=(0, 255, 255)):
        """
        Draw animated ball trajectories from players to landing positions and continue after bounce to next landing.
        Trajectories gradually appear as frames progress between hits, landings and next hits/landings.
        When a new shot is hit, previous complete trajectories are removed.
        
        Args:
            frames (list): List of frames to draw the trajectories on
            player_mini_court_detections (list): List of dictionaries containing player positions in mini court
            ball_mini_court_detections (list): List of dictionaries containing ball positions in mini court
            ball_landing_frames (list): List of frames where the ball lands
            ball_shots_frames (list): List of frames where players hit the ball
            serve_player (str): Which player serves first - 'Lower' or 'Upper'
            line_color (tuple): Color for trajectory line (BGR format)

        Returns:
            list: List of frames with trajectories applied
        """
        output_frames = frames.copy()
        
        # Get player IDs based on court position
        player_ids = {}
        if len(player_mini_court_detections) > 0 and len(player_mini_court_detections[0]) >= 2:
            # Get the first frame with both players
            for frame_idx, players in enumerate(player_mini_court_detections):
                if len(players) >= 2:
                    player_y_values = {}
                    for player_id, pos in players.items():
                        player_y_values[player_id] = pos[1]
                    
                    # Sort by Y value (higher Y is lower court)
                    sorted_players = sorted(player_y_values.items(), key=lambda x: x[1])
                    
                    # First player (smaller Y) is Upper, second is Lower
                    if len(sorted_players) >= 2:
                        player_ids['Upper'] = sorted_players[0][0]
                        player_ids['Lower'] = sorted_players[1][0]
                        break
        
        # Set initial player based on serve_player
        current_player = serve_player
        
        # Ensure we have valid data to draw trajectories
        if (not ball_shots_frames or not ball_landing_frames or
            len(ball_shots_frames) == 0 or len(ball_landing_frames) == 0 or
            len(player_ids) < 2):
            return output_frames
        
        # Organize all events chronologically
        all_events = []
        
        # Add hit events
        for hit_frame in ball_shots_frames:
            all_events.append((hit_frame, 'hit'))
        
        # Add landing events
        for landing_frame in ball_landing_frames:
            all_events.append((landing_frame, 'landing'))
        
        # Sort events chronologically
        all_events.sort(key=lambda x: x[0])
        
        # Create a list of trajectories to draw
        trajectories = []
        
        # Starting point is the first hit
        current_start_frame = None
        current_start_type = None
        
        # Process all events to create trajectory segments
        for i, (frame, event_type) in enumerate(all_events):
            # Skip if this is the first event (need a starting point)
            if current_start_frame is None:
                current_start_frame = frame
                current_start_type = event_type
                continue
                
            # We have a start and end point for a trajectory segment
            trajectories.append({
                'start_frame': current_start_frame,
                'start_type': current_start_type,
                'end_frame': frame,
                'end_type': event_type
            })
            
            # Update start for next segment
            current_start_frame = frame
            current_start_type = event_type
        
        # Keep track of previous trajectory points for continuous tracing
        prev_start_pos = None
        prev_end_pos = None
        
        # Draw yellow dots at bounce points
        for landing_frame in ball_landing_frames:
            if landing_frame < len(output_frames) and 1 in ball_mini_court_detections[landing_frame]:
                ball_pos = ball_mini_court_detections[landing_frame][1]
                x, y = int(ball_pos[0]), int(ball_pos[1])
                # Draw a larger yellow dot at each bounce point
                cv2.circle(output_frames[landing_frame], (x, y), 8, (0, 255, 255), -1)
        
        # Dictionary to store completed trajectory segments for each frame
        completed_trajectories = {i: [] for i in range(len(output_frames))}
        
        # Track the latest hit frame for clearing previous trajectories
        latest_hit_frame = -1
        
        # Process each trajectory
        for trajectory_idx, trajectory in enumerate(trajectories):
            start_frame = trajectory['start_frame']
            end_frame = trajectory['end_frame']
            start_type = trajectory['start_type']
            end_type = trajectory['end_type']
            
            # If this is a new hit, clear the previous trajectories
            if start_type == 'hit' and start_frame > latest_hit_frame:
                latest_hit_frame = start_frame
                # Clear previous trajectories from this point onwards
                for frame_idx in range(start_frame, len(output_frames)):
                    completed_trajectories[frame_idx] = []
            
            # Get starting position based on event type
            if start_type == 'hit':
                # If hit, get player position
                current_player_id = player_ids.get(current_player)
                if current_player_id not in player_mini_court_detections[start_frame]:
                    # Switch player for next segment if needed
                    if end_type == 'hit':
                        current_player = 'Upper' if current_player == 'Lower' else 'Lower'
                    continue
                
                start_pos = player_mini_court_detections[start_frame][current_player_id]
            else:  # landing
                # If landing, get ball position
                if 1 not in ball_mini_court_detections[start_frame]:
                    continue
                start_pos = ball_mini_court_detections[start_frame][1]
            
            # Get ending position based on event type
            if end_type == 'hit':
                # Switch player since ball is being hit by other player
                current_player = 'Upper' if current_player == 'Lower' else 'Lower'
                current_player_id = player_ids.get(current_player)
                
                if current_player_id not in player_mini_court_detections[end_frame]:
                    continue
                
                end_pos = player_mini_court_detections[end_frame][current_player_id]
            else:  # landing
                if 1 not in ball_mini_court_detections[end_frame]:
                    continue
                end_pos = ball_mini_court_detections[end_frame][1]
            
            # If this is a trajectory after a bounce, calculate a natural bounce trajectory
            # instead of drawing directly to the player
            if start_type == 'landing' and prev_start_pos is not None and prev_end_pos is not None:
                # Calculate the direction vector of the previous trajectory
                prev_direction_x = prev_end_pos[0] - prev_start_pos[0]
                prev_direction_y = prev_end_pos[1] - prev_start_pos[1]
                
                # Scale down the direction vector to create a bounce effect
                bounce_direction_x = prev_direction_x * 0.25
                bounce_direction_y = prev_direction_y * 0.25
                
                # Calculate a virtual end point that creates a realistic arc
                virtual_end_x = start_pos[0] + bounce_direction_x
                virtual_end_y = start_pos[1] + bounce_direction_y
                
                # Use this virtual end point for a natural bounce trajectory
                # instead of pointing directly to the player
                end_pos = (virtual_end_x, virtual_end_y)
            
            start_x, start_y = int(start_pos[0]), int(start_pos[1])
            end_x, end_y = int(end_pos[0]), int(end_pos[1])
            
            # Save positions for next trajectory segment
            prev_start_pos = start_pos
            prev_end_pos = end_pos
            
            # Draw the gradually appearing trajectory
            for frame_idx in range(start_frame, min(end_frame + 1, len(output_frames))):
                # Calculate how much of the trajectory to draw (0.0 to 1.0)
                if end_frame == start_frame:
                    progress = 1.0
                else:
                    progress = (frame_idx - start_frame) / (end_frame - start_frame)
                
                # Calculate the endpoint of the partial trajectory
                current_end_x = int(start_x + (end_x - start_x) * progress)
                current_end_y = int(start_y + (end_y - start_y) * progress)
                
                # Add this segment to the current frame's completed trajectories once it's fully drawn
                if progress == 1.0:
                    completed_trajectories[frame_idx].append({
                        'start': (start_x, start_y),
                        'end': (current_end_x, current_end_y),
                        'is_hit': start_type == 'hit'
                    })
                
                # Draw all completed trajectories for this frame first
                for traj in completed_trajectories[frame_idx]:
                    traj_start_x, traj_start_y = traj['start']
                    traj_end_x, traj_end_y = traj['end']
                    
                    # Draw full trajectory as dotted line
                    line_length = np.sqrt((traj_end_x - traj_start_x)**2 + (traj_end_y - traj_start_y)**2)
                    num_segments = max(2, int(line_length / 10))  # At least 2 segments
                    
                    # Draw each segment
                    for i in range(num_segments):
                        t = i / (num_segments - 1)
                        dot_x = int(traj_start_x + t * (traj_end_x - traj_start_x))
                        dot_y = int(traj_start_y + t * (traj_end_y - traj_start_y))
                        
                        # Draw a circle at this position (only every other segment for dotted effect)
                        if i % 2 == 0:
                            cv2.circle(output_frames[frame_idx], (dot_x, dot_y), 3, line_color, -1)
                    
                    # Draw endpoint with bigger circle if this was a hit
                    if traj['is_hit']:
                        cv2.circle(output_frames[frame_idx], (traj_end_x, traj_end_y), 5, line_color, -1)
                
                # Now draw the current trajectory segment that's being animated
                # Create dotted line effect
                line_length = np.sqrt((current_end_x - start_x)**2 + (current_end_y - start_y)**2)
                num_segments = max(2, int(line_length / 10))  # At least 2 segments
                
                # Draw each segment
                for i in range(num_segments):
                    t = i / (num_segments - 1)
                    dot_x = int(start_x + t * (current_end_x - start_x))
                    dot_y = int(start_y + t * (current_end_y - start_y))
                    
                    # Draw a circle at this position (only every other segment for dotted effect)
                    if i % 2 == 0:
                        dot_size = max(1, int(3 * progress))
                        cv2.circle(output_frames[frame_idx], (dot_x, dot_y), dot_size, line_color, -1)
                
                # Draw the end point when we're near the end
                if progress > 0.85 and start_type == 'hit':
                    cv2.circle(output_frames[frame_idx], (end_x, end_y), 5, line_color, -1)
                    
            # After completing a trajectory, add it to all future frames until next hit
            for future_frame in range(end_frame + 1, len(output_frames)):
                # Break if we reach a new hit frame that's not directly connected to this trajectory
                if future_frame > end_frame and future_frame in ball_shots_frames and not (end_type == 'landing' and future_frame == ball_shots_frames[0]):
                    break
                    
                completed_trajectories[future_frame].append({
                    'start': (start_x, start_y),
                    'end': (end_x, end_y),
                    'is_hit': start_type == 'hit'
                })
            
            # Only switch player after hit-landing-hit sequence
            if start_type == 'hit' and end_type == 'landing':
                # Don't switch player here, wait for next hit
                pass
            elif start_type == 'landing' and end_type == 'hit':
                # Don't need to switch here as we already did above
                pass
        
        return output_frames
                
    
    
    def draw_player_distance_heatmap(self, frames, player_mini_court_detections, color_map=cv2.COLORMAP_HOT, selected_player='Lower', alpha=0.6):
        """
        Draw a heatmap representing opponent player distance on the mini court.
        
        Args:
            frames (list): List of frames to draw the heatmap on
            player_mini_court_detections (list): List of dictionaries containing player positions in mini court
            color_map: OpenCV colormap to use (default: cv2.COLORMAP_HOT)
            selected_player (str): Which part of the court to show - 'Lower' or 'Upper'
            alpha: Transparency level for the heatmap overlay (default: 0.6)
        
        Returns:
            list: List of frames with heatmap applied
        """
        output_frames = []

        
        # Get the net y-coordinate (dividing line between upper and lower court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5])/2)
        
        # Use actual tennis court boundaries from keypoints
        court_left_x = int(self.drawing_key_points[0])  # Left boundary from top-left corner
        court_right_x = int(self.drawing_key_points[2])  # Right boundary from top-right corner
        court_top_y = int(self.drawing_key_points[1])    # Top boundary from top-left corner
        court_bottom_y = int(self.drawing_key_points[5])  # Bottom boundary from bottom-left corner
        
        # Define court region based on selected_player
        if selected_player == 'Lower':
            # If Lower selected, show heatmap in Upper court
            court_y_start = court_top_y
            court_y_end = net_y
        else:
            # If Upper selected, show heatmap in Lower court
            court_y_start = net_y
            court_y_end = court_bottom_y
        
        # Court dimensions
        court_width = court_right_x - court_left_x
        court_height = court_y_end - court_y_start
        
        # Maximum possible distance for normalization
        max_distance = np.sqrt(court_width**2 + court_height**2)
        
        for frame_num, frame in enumerate(frames):
            # Create a copy of the frame
            heatmap_frame = frame.copy()
            
            # Get player positions for this frame
            player_positions = player_mini_court_detections[frame_num]
            
            # Create a blank image for the heatmap (same size as the court section)
            player_img = np.zeros((court_height, court_width, 3), dtype=np.uint8)
            
            # Process each player position
            for player_id, position in player_positions.items():
                x, y = position
                
                # Check if the player is inside or near the selected court area
                if (y >= court_y_start - 100 and y <= court_y_end + 100):
                    # Adjust player coordinates to be relative to the heatmap
                    rel_x = int(x) - court_left_x
                    rel_y = int(y) - court_y_start
                    
                    # Create coordinate grids for the court area
                    y_coords, x_coords = np.ogrid[:court_height, :court_width]
                    
                    # Calculate Euclidean distance using vectorization
                    player_distance = np.sqrt((x_coords - rel_x)**2 + (y_coords - rel_y)**2)
                    
                    # Clip to max distance
                    player_distance = np.clip(player_distance, 0, max_distance)
                    
                    # Normalize distance to range 0-255
                    player_intensity = 255 * player_distance / max_distance
                    

                    player_img[:, :, 0] = player_intensity
                    player_img[:, :, 1] = player_intensity
                    player_img[:, :, 2] = player_intensity
                    
                    #TODO: Convert to heatmap values (0-1) then back to pixel values and return the heatmap image
                    #player_heatmap_frame = convert_to_heatmap_values(player_img)

                    
            
            # Apply the colormap to the final heatmap
            colored_heatmap = cv2.applyColorMap(player_img, color_map)
            
            # Create a temporary frame with the heatmap region
            overlay = heatmap_frame.copy()
            
            # Place the colored heatmap onto the selected court region
            overlay[court_y_start:court_y_end, court_left_x:court_right_x] = colored_heatmap
            
            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, alpha, heatmap_frame, 1-alpha, 0, heatmap_frame)
            
            output_frames.append(heatmap_frame)
        
        return output_frames
    
    def draw_ball_landing_heatmap(self, frames, player_balls_frames, ball_mini_court_detections, ball_shots_frames_upper, ball_shots_frames_lower, selected_player, radius=20, color_map=cv2.COLORMAP_HOT, alpha=0.6):
        """
        Draw a heatmap representing ball landing area on the opponent mini court at each shot of the selected player.
        
        Args:
            frames (list): List of frames to draw the heatmap on
            player_balls_frames (list): List of frames with ball landing positions in the opposite court 
            ball_mini_court_detections (list): List of dictionaries containing ball positions in mini court
            ball_shots_frames_upper (list): List of frames where upper player hits the ball
            ball_shots_frames_lower (list): List of frames where lower player hits the ball
            selected_player (str): Which part of the court to show - 'Lower' or 'Upper'
            radius (int): Radius of the ball landing area
            color_map: OpenCV colormap to use (default: cv2.COLORMAP_HOT)
            alpha: Transparency level for the heatmap overlay (default: 0.6)
        
        Returns:
            list: List of frames with heatmap applied
        """
        output_frames = []

        # Get the net y-coordinate (dividing line between upper and lower court)
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)

        # Use actual tennis court boundaries from keypoints
        court_left_x = int(self.drawing_key_points[0])  # Left boundary from top-left corner
        court_right_x = int(self.drawing_key_points[2])  # Right boundary from top-right corner
        court_top_y = int(self.drawing_key_points[1])    # Top boundary from top-left corner
        court_bottom_y = int(self.drawing_key_points[5])  # Bottom boundary from bottom-left corner

        # Define court region based on selected_player
        if selected_player == 'Lower':
            # If Lower selected, show heatmap in Upper court
            court_y_start = court_top_y
            court_y_end = net_y
            
            # Draw heatmap from the Lower player shots to the Upper player shots
            shots_starting_frames = ball_shots_frames_lower
            shots_ending_frames = ball_shots_frames_upper
        else:
            # If Upper selected, show heatmap in Lower court
            court_y_start = net_y
            court_y_end = court_bottom_y
            
            # Draw heatmap from the Upper player shots to the Lower player shots
            shots_starting_frames = ball_shots_frames_upper
            shots_ending_frames = ball_shots_frames_lower

        # Court dimensions
        court_width = court_right_x - court_left_x
        court_height = court_y_end - court_y_start

        # Maximum possible distance for normalization
        max_distance = np.sqrt(court_width**2 + court_height**2)

        for frame_num, frame in enumerate(frames):
            # Create a copy of the frame
            heatmap_frame = frame.copy()

            # Initialize a blank heatmap image
            heatmap_img = np.zeros((court_height, court_width, 3), dtype=np.uint8)

            # Check if the current frame is a ball landing frame
            if frame_num in player_balls_frames:
                
                # Get ball positions for this frame
                ball_positions = ball_mini_court_detections[frame_num]
                
                for _, position in ball_positions.items():
                    # Select the ball coordinates
                    x, y = position

                    # Check if the ball is inside or near the selected court area
                    if court_y_start <= y <= court_y_end:
                        # Adjust ball coordinates to be relative to the heatmap
                        rel_x = int(x) - court_left_x
                        rel_y = int(y) - court_y_start

                        # Create coordinate grids for the court area
                        y_coords, x_coords = np.ogrid[:court_height, :court_width]

                        # Calculate Euclidean distance using vectorization
                        ball_distance = np.sqrt((x_coords - rel_x)**2 + (y_coords - rel_y)**2)

                        # Clip to max distance
                        ball_distance = np.clip(ball_distance, 0, max_distance)

                        # Normalize distance to range 0-255
                        ball_intensity = 255 * ball_distance / max_distance

                        # Create a mask for the ball's distance using ball landing area
                        distance_mask = ball_distance >= radius

                        # Update the heatmap image with the ball intensity
                        heatmap_img[:, :, 0] = ball_intensity
                        heatmap_img[:, :, 1] = ball_intensity
                        heatmap_img[:, :, 2] = ball_intensity

                        # Invert the intensity
                        heatmap_img = 255 - heatmap_img

                        # Apply the mask to the ball image
                        heatmap_img[distance_mask] = 0


            # Apply the colormap to the final heatmap
            colored_heatmap = cv2.applyColorMap(heatmap_img, color_map)

            # Create a temporary frame with the heatmap region
            overlay = heatmap_frame.copy()

            # Place the colored heatmap onto the selected court region
            overlay[court_y_start:court_y_end, court_left_x:court_right_x] = colored_heatmap

            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, alpha, heatmap_frame, 1 - alpha, 0, heatmap_frame)

            output_frames.append(heatmap_frame)

        return output_frames