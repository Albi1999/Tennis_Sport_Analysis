import cv2
import numpy as np
import sys
sys.path.append('../')
import info
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_center_of_bbox,
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250          # Width of the mini court in pixels
        self.drawing_rectangle_height = 500         # Height of the mini court in pixels
        self.buffer = 50                            # Distance from the right and top of the frame to the mini court
        self.padding_court=20                       # Padding from the mini court to the court 

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

    def draw_court(self,frame):
        '''Draw the mini court on the frame'''
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,255,0),-1)

        # Draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        '''Draw the background rectangle on the frame'''
        shapes = np.zeros_like(frame,np.uint8) 
        alpha = 0 # Transparency factor
        
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

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
