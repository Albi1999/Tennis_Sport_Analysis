from ultralytics import YOLO
import supervision as sv
import cv2
import pickle 
import numpy as np
import sys 
sys.path.append('../') # go out of the trackers folder 
from utils import get_center_of_bbox, euclidean_distance


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Filter found objects of class 'person' by choosing only the 2 actual players
        based on the first frame.

        Args:
            court_keypoints (List) : List of keypoints of the court.
            player_detections (List of Dicts) : List of Dicts, where each Dict contains
            the found players in a frame, given by their ID (key) and bounding box coordinates (value).

        Returns:
            filtered_player_detections (List of Dicts) : List of Dicts, where each Dict contains
            the found players in a frame, given by their ID (key) and bounding box coordinates (value).
            chosen_players_ids (List) : the tracking IDs of the 2 players.
        """
        
        filtered_player_detections = []
        
        # Collect the movements of all detected persons
        persons_movement = {}
        for frame_num, player_dict in enumerate(player_detections):
            for track_id, bbox in player_dict.items():
                player_center = get_center_of_bbox(bbox)
                if track_id in persons_movement:
                    persons_movement[track_id].append(player_center)
                else: 
                    persons_movement[track_id] = [player_center]

        # Calculate total displacement and average displacement per frame
        persons_displacement = {}
        persons_consistency = {}  # Track how consistently the player appears
        
        for track_id, movements in persons_movement.items():
            if len(movements) < 5:  # Skip players that appear in very few frames
                continue
                
            # Calculate total displacement
            total_displacement = 0
            for i in range(1, len(movements)):
                prev_pos = movements[i-1]
                curr_pos = movements[i]
                displacement = euclidean_distance(prev_pos, curr_pos)
                total_displacement += displacement
                
            # Calculate average displacement and consistency score
            avg_displacement = total_displacement / len(movements) if len(movements) > 0 else 0
            consistency = len(movements) / len(player_detections)  # Ratio of frames where player appears
            
            persons_displacement[track_id] = avg_displacement * consistency  # Weight by consistency
            persons_consistency[track_id] = consistency

        # Sort by weighted displacement (higher displacement and more consistent appearance)
        if len(persons_displacement) >= 2:
            print("Choosing players based on displacement")
            persons_displacement = dict(sorted(persons_displacement.items(), key=lambda x: x[1], reverse=True))
            
            # Gather the two persons with highest weighted displacement
            chosen_players_ids = list(persons_displacement.keys())[:2]
            
            # Get their bounding boxes
            for player_dict in player_detections:
                filtered_player_dict = {}
                for track_id, bbox in player_dict.items():
                    if track_id in chosen_players_ids:
                        filtered_player_dict[track_id] = bbox
                filtered_player_detections.append(filtered_player_dict)
                
            return filtered_player_detections, chosen_players_ids
        else:
            print("Choosing players based on keypoints")
            # Fallback to the closest-to-keypoints method if not enough moving players
            player_detections_first_frame = player_detections[0]
            chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
            filtered_player_detections = []
            
            for player_dict in player_detections: 
                filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
                filtered_player_detections.append(filtered_player_dict)
                
            return filtered_player_detections, chosen_players



    
    def choose_players(self, court_keypoints, player_dict):
        """
        Choose the two players that are playing against each other
        by iterating over all found objects of the person class in 
        the first frame and then selecting the 2 persons that are
        closest to any of the keypoints.

        Note: This needs an initial frame where the 2 players are the
        closest to any of the keypoints ; i.e. nobody else can be on the
        playing field or extremely close to it!

        Args:
            ...

        Returns:
            chosen_players (List) : the tracking IDs of the 2 players.
        """
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox) 

            min_distance = float('inf')
         
            # Find the closest keypoint to the selected person (i.e. player_dict still
            # contains all persons that were detected)
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = euclidean_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance 
            distances.append((track_id, min_distance))

        distances = sorted(distances, key = lambda x : x[1])

        # Choose the 2 players (i.e. the 2 smallest distances) by their ID
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players








    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        """
        Iterate over all the frames of the video, and
        collect the found players.

        Args:
            frames : All the frames of the video.
            read_from_stub (bool) : Store player detections. 
            stub_path (str) : Path to store player detections in.

        Returns:
            player_detections (List of Dicts) : List 
            of the Dicts generated by 'detect_frame'.
        """
        player_detections = []

        # Load stored detected player informations
        if stub_path is not None and read_from_stub:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections 

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # Store detected player information
        if stub_path is not None: 
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)


        return player_detections 
    
    def detect_frame(self, frame):
        """
        Frame by frame, detect the players and track them

        Args:
            frame : the frame to be analyzed 

        Returns: 
            player_dict (dict) : Dictionary of the found players,
                                 given by their id (key) and 
                                 bounding box coordinates (value).     
        """
        # Set persist to True, such that the model recognizes that it will get multiple frame inputs,
        # one after the other and should be persistent (i.e. track over multiple frames)
        results = self.model.track(frame, persist = True)[0]

        # Stores tracked IDs to class names in a dictionary
        id_name_dict = results.names

        # Store players with their bounding boxes, i.e.
        # key : ID, value : bounding box (x_min,y_min,x_max,y_max)
        player_dict = {}

        # Iterate over all found boxes 
        for box in results.boxes:
            # Get the id of the current object 
            track_id = int(box.id.tolist()[0])
            # Get the bounding box (x_min,y_min,x_max,y_max)
            result = box.xyxy.tolist()[0]
            # Get class ID of object
            object_cls_id = box.cls.tolist()[0]
            # Find the corresponding class name of the object, given the class ID
            object_cls_name = id_name_dict[object_cls_id]
            # We are only tracking persons, therefore filter for these
            if object_cls_name == 'person':
                player_dict[track_id] = result

        return player_dict 
    

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draw bounding boxes as well as IDs for the players.

        Args:
            video_frames : List of all the frames of the video.
            player_detections : List returned by 'detect_frames()'.
        
        Returns:
            output_video_frames : Frames of the video, now annotated
                                  with bounding boxes & IDs for Players.
        
        
        """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes 
            for track_id, bbox in player_dict.items():
                # extract coordinates of the bounding box
                x1, y1, x2, y2 = bbox
                # Draw text indicating the Player ID (Cyan)
                cv2.putText(frame, f"Player ID : {track_id}", (int(x1), int(y1) - 5),cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 2)
                # Draw the rectangle onto the frame (Cyan)
                cv2.rectangle(frame , (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)

            output_video_frames.append(frame)
        
        return output_video_frames

    # TODO : Implement this method with supervision library
    def draw_ellipse_bboxes(self, video_frames, player_detections, player):
        """
        Draw ellipses around the players in cyan color.
        
        Args:
            video_frames (list): List of frame images.
            player_detections (list): List of dictionaries, where each dictionary contains the player detections in a frame.
            player (str): 'Upper' or 'Lower' player.
        
        Returns:
            output_video_frames (list): Frames of the video, now annotated with ellipses around players.
        """
        output_video_frames = []
        
        # Create an EllipseAnnotator object with custom colors
        color_palette = ['#00ffff', '#800080'] # Cyan and Purple
        color_idx = 0 if player == 'Upper' else 1
        ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(color_palette))
        triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex(color_palette[color_idx]))

        for frame, player_dict in zip(video_frames, player_detections):
            # Make a copy of the original frame
            
            if player_dict:  # Check if there are any detections
                # Collect all bounding boxes and IDs
                xyxy_list = []
                class_id_list = []
                
                for track_id, bbox in player_dict.items():
                    xyxy_list.append(bbox)
                    class_id_list.append(track_id)
                
                # Convert to numpy arrays
                xyxy_array = np.array(xyxy_list)
                class_id_array = np.array(class_id_list)
                
                # Create detections object with all players
                detections = sv.Detections(
                    xyxy=xyxy_array,
                    class_id=class_id_array
                )
                
                selected_player_detections = sv.Detections.empty()  # Inizializza come vuoto

                if len(xyxy_array) >= 2:
                    # Calcola il centro y di ciascun bbox per determinare quale è "Upper" e quale è "Lower"
                    y_centers = [(bbox[1] + bbox[3]) / 2 for bbox in xyxy_array]
                    
                    # Seleziona l'indice del giocatore in base alla posizione
                    if player == 'Upper':
                        selected_index = np.argmin(y_centers)  # Il giocatore con y minore è in alto
                    else:  # 'Lower'
                        selected_index = np.argmax(y_centers)  # Il giocatore con y maggiore è in basso
                    
                    # Crea detections solo per il giocatore selezionato
                    selected_player_detections = sv.Detections(
                        xyxy=np.array([xyxy_array[selected_index]]),
                        class_id=np.array([class_id_array[selected_index]])
                    )
                    
                elif len(xyxy_array) == 1:
                    # Se c'è un solo giocatore rilevato, usa quello
                    selected_player_detections = detections
            
                
                # Annotate the frame with ellipses and triangles
                annotated_frame = ellipse_annotator.annotate(frame, detections)
                annotated_frame = triangle_annotator.annotate(frame, selected_player_detections)
            
            output_video_frames.append(annotated_frame)
        
        return output_video_frames


