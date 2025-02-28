import cv2
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
        self.drawing_rectangle_width = 250          # Width of the mini court in pixels
        self.drawing_rectangle_height = 500         # Height of the mini court in pixels
        self.buffer = 50                            # Distance from the right and top of the frame to the mini court
        self.padding_court= 20                      # Padding from the mini court to the court 

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


    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes, ball_boxes, original_court_key_points, chosen_players_ids ):
        """ Convert the bounding boxes of the players and the ball to the mini court coordinates.
        Args:
            player_boxes (list): List of dictionaries containing the bounding boxes of the players.
            ball_boxes (list): List of dictionaries containing the bounding boxes of the ball.
            original_court_key_points (list): List of the key points of the court in the original frame.
        Returns:
            output_player_boxes (list): List of dictionaries containing the bounding boxes of the players in the mini court.
            output_ball_boxes (list): List of dictionaries containing the bounding boxes of the ball in the mini court.
        """

        # TODO : fix the player_heights (bc we dont know which player is which rn) ;
        # TODO : fix that when a player is not tracked all the way through (all frames), this here fails


        player_heights = {
            chosen_players_ids[0]: info.PLAYER_1_HEIGHT_METERS,
            chosen_players_ids[1]: info.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes= []
        output_ball_boxes= []


        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            
            if player_bbox:
                closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: euclidean_distance(ball_position, get_center_of_bbox(player_bbox[x])))
            else:
                closest_player_id_to_ball = None
            
            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                
                # Get the maximum height of the player in pixels
                bboxes_heights_in_pixels = []
                for i in range(frame_index_min, frame_index_max):
                    if player_id in player_boxes[i]:
                        bboxes_heights_in_pixels.append(get_height_of_bbox(player_boxes[i][player_id]))

                if bboxes_heights_in_pixels:
                    max_player_height_in_pixels = max(bboxes_heights_in_pixels)
                else:
                    # Default value
                    max_player_height_in_pixels = 200  # value to be adjusted


                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get the closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes
    
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
    
    
    def create_heatmap_animation(self, player_mini_court_detections, output_path="/output/heatmap_animation.mp4", 
                            resolution=20, color_map=cv2.COLORMAP_HOT, alpha=0.6, fps=15):
        """
        Creates an animation showing only the tennis court and dynamic heatmap.
        
        Parameters:
        player_mini_court_detections - Player positions for each frame
        output_path - Path where to save the animation
        resolution - Grid resolution for the heatmap
        color_map - OpenCV color map to use
        alpha - Heatmap transparency
        fps - Frames per second of the animation
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

        # Create a video writer
        width = self.drawing_rectangle_width
        height = self.drawing_rectangle_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # For each frame, create a specific heatmap
        for frame_idx, frame_player_positions in enumerate(player_mini_court_detections):
            # Create an empty black image
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

            # Create the grid for this frame's heatmap
            heatmap_grid = np.zeros((grid_height, grid_width))

            # For each grid point, calculate distance from nearest player
            for i in range(grid_width):
                for j in range(grid_height):
                    grid_x = self.court_start_x + i * x_step
                    grid_y = self.court_start_y + j * y_step

                    # Apply heatmap only to upper part of court (above net)
                    if grid_y >= net_y:
                        continue  # Skip pixels in lower part

                    min_distance = max_distance
                    for player_id, position in frame_player_positions.items():
                        if position is None:
                            continue

                        player_x, player_y = position
                        distance = euclidean_distance((grid_x, grid_y), (player_x, player_y))
                        min_distance = min(min_distance, distance)

                    # Normalize distance between 0 and 1
                    normalized_distance = min_distance / max_distance
                    heatmap_grid[j, i] = normalized_distance

            # Apply color map to grid
            heatmap_colored = cv2.applyColorMap((heatmap_grid * 255).astype(np.uint8), color_map)

            # Resize to court dimensions
            heatmap_resized = cv2.resize(
                heatmap_colored, 
                (self.court_drawing_width, self.court_end_y - self.court_start_y)
            )

            # Create transparent overlay of heatmap
            overlay = court_image.copy()

            # Apply heatmap only to upper part of court
            court_y_offset = self.padding_court
            court_x_offset = self.padding_court
            
            overlay[
                court_y_offset:net_y-self.start_y,
                court_x_offset:self.court_drawing_width+court_x_offset
            ] = heatmap_resized[0:(net_y-self.court_start_y), :]

            # Blend overlay with court image
            result = cv2.addWeighted(overlay, alpha, court_image, 1-alpha, 0)
            
            # Draw players as points
            for player_id, position in frame_player_positions.items():
                if position is None:
                    continue
                x, y = position
                x_adj = int(x - self.start_x)
                y_adj = int(y - self.start_y)
                if 0 <= x_adj < width and 0 <= y_adj < height:
                    cv2.circle(result, (x_adj, y_adj), 5, (0, 0, 255), -1)  # Red for players

            # Write frame to video
            video_writer.write(result)

        # Release the writer
        video_writer.release()
        
        return output_path