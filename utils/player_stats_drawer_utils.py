import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats, selected_player=None, player_mapping=None):
    """
    Draws the player statistics on the output video frames.
    
    Args:
        output_video_frames: List of frames of the output video.
        player_stats: DataFrame containing player statistics.
        selected_player: 'Upper' or 'Lower' if only displaying one player's stats, None for both
        player_mapping: Dictionary mapping 'Upper'/'Lower' to player IDs
    
    Returns:
        List of frames with player statistics drawn on them.
    """
    # Determine if we're showing stats for a specific player or both
    show_both_players = (selected_player is None or player_mapping is None)
    
    # If we're only showing one player, get their player ID
    selected_player_id = None
    if not show_both_players and selected_player in player_mapping:
        selected_player_id = player_mapping[selected_player]

    for index, row in player_stats.iterrows():
        frame = output_video_frames[index]
        
        # Add player stats to the frame
        shapes = np.zeros_like(frame, np.uint8)
        # Dimensions of the rectangle
        width = 330
        height = 380 if not show_both_players else 270  # Increased height for additional stats
        # Position of the rectangle
        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - (height + 50)
        end_x = start_x + width
        end_y = start_y + height
        
        # Draw the rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add the header text
        if show_both_players:
            text = "     Player 1     Player 2"
            cv2.putText(frame, text, (start_x+80, start_y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Determine the color based on the selected player
            if selected_player == 'Upper':
                player_color = (255, 255, 0)  # Cyan
            else:
                player_color = (255, 0, 255)  # Magenta
        
            # Only show selected player
            text = f"{selected_player} Player Stats"
            cv2.putText(frame, text, (start_x+50, start_y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, player_color, 2)

            
        # Add stats rows
        if show_both_players:
            # Shot speed
            text = "Shot Speed"
            cv2.putText(frame, text, (start_x+10, start_y+80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_last_shot_speed']:.1f} km/h    {row['player_2_last_shot_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+130, start_y+80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Average shot speed
            text = "avg. Shot Speed"
            cv2.putText(frame, text, (start_x+10, start_y+120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_average_shot_speed']:.1f} km/h    {row['player_2_average_shot_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+130, start_y+160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Player speed
            text = "Player Speed"
            cv2.putText(frame, text, (start_x+10, start_y+160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_last_player_speed']:.1f} km/h    {row['player_2_last_player_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+130, start_y+120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
            # Average player speed
            text = "avg. Player Speed"
            cv2.putText(frame, text, (start_x+10, start_y+200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_average_player_speed']:.1f} km/h    {row['player_2_average_player_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+130, start_y+200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        else:
            # Only show selected player stats with larger font and cleaner layout
            y_offset = start_y + 80
            line_spacing = 35
            
            # Player speed
            text = "Player Speed"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_last_player_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            

            # Average player speed
            y_offset += line_spacing
            text = "Avg Player Speed"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_average_player_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Shot speed
            y_offset += line_spacing
            text = "Shot Speed"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_last_shot_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Min shot speed
            y_offset += line_spacing
            text = "Min Shot Speed"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Handle case where no shots were made
            min_shot_speed = row[f'player_{selected_player_id}_min_shot_speed']
            if min_shot_speed == float('inf'):
                text = "N/A"
            else:
                text = f"{min_shot_speed:.1f} km/h"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Max shot speed
            y_offset += line_spacing
            text = "Max Shot Speed"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_max_shot_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Average shot speed
            y_offset += line_spacing
            text = "Avg Shot Speed"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_average_shot_speed']:.1f} km/h"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        
            # Hits counter
            y_offset += line_spacing
            text = "Hits Counter"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{int(row[f'player_{selected_player_id}_hits_counter'])}"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Distance covered
            y_offset += line_spacing
            text = "Distance Covered"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_distance_covered']:.1f} m"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Score probability
            y_offset += line_spacing
            text = "Score Probability"
            cv2.putText(frame, text, (start_x+10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_score_probability']:.1f}%"
            cv2.putText(frame, text, (start_x+180, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        output_video_frames[index] = frame
    
    return output_video_frames

def create_player_stats_box_video(player_stats, video_number, fps=30, selected_player=None, player_mapping=None):
    """
    Creates a separate video containing only the player stats box.
    
    Args:
        player_stats: DataFrame containing player statistics.
        video_number: Video identifier.
        fps: Frames per second for the output video.
        selected_player: 'Upper' or 'Lower' if only displaying one player's stats, None for both
        player_mapping: Dictionary mapping 'Upper'/'Lower' to player IDs
    
    Returns: 
        Path to the saved stats box video.
    """
    # Determine if we're showing stats for a specific player or both
    show_both_players = (selected_player is None or player_mapping is None)
    
    # If we're only showing one player, get their player ID
    selected_player_id = None
    if not show_both_players and selected_player in player_mapping:
        selected_player_id = player_mapping[selected_player]
    
    width = 380
    height = 380 if not show_both_players else 270
    output_path = f"output/animations/player_stats_box{video_number}.mp4"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Iterate over each frame in player_stats
    for _, row in player_stats.iterrows():
        # Create a blank frame for each stats update
        stats_box_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Header positioning
        if show_both_players:
            text = "     Player 1     Player 2"
            cv2.putText(stats_box_frame, text, (width//8, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Stats lines
            y_offset = 80
            line_spacing = 40
            
            # Shot speed
            text = "Shot Speed"
            cv2.putText(stats_box_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row['player_1_last_shot_speed']:.1f} km/h    {row['player_2_last_shot_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (130, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Player speed
            y_offset += line_spacing
            text = "Player Speed"
            cv2.putText(stats_box_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row['player_1_last_player_speed']:.1f} km/h    {row['player_2_last_player_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (130, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Average shot speed
            y_offset += line_spacing
            text = "Avg Shot Speed"
            cv2.putText(stats_box_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row['player_1_average_shot_speed']:.1f} km/h    {row['player_2_average_shot_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (130, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Average player speed
            y_offset += line_spacing
            text = "Avg Player Speed"
            cv2.putText(stats_box_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text = f"{row['player_1_average_player_speed']:.1f} km/h    {row['player_2_average_player_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (130, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        else:
            # Only show selected player with cleaner, more prominent display
            text = f"{selected_player} Player Stats"
            cv2.putText(stats_box_frame, text, (width//8, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Stats with larger font and better spacing
            y_offset = 80
            line_spacing = 35
            
            # Shot speed
            text = "Shot Speed"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_last_shot_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Min shot speed
            y_offset += line_spacing
            text = "Min Shot Speed"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Handle case where no shots were made
            min_shot_speed = row[f'player_{selected_player_id}_min_shot_speed']
            if min_shot_speed == float('inf'):
                text = "N/A"
            else:
                text = f"{min_shot_speed:.1f} km/h"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Max shot speed
            y_offset += line_spacing
            text = "Max Shot Speed"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_max_shot_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Player speed
            y_offset += line_spacing
            text = "Player Speed"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_last_player_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Average shot speed
            y_offset += line_spacing
            text = "Avg Shot Speed"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_average_shot_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Average player speed
            y_offset += line_spacing
            text = "Avg Player Speed"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_average_player_speed']:.1f} km/h"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hits counter
            y_offset += line_spacing
            text = "Hits Counter"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{int(row[f'player_{selected_player_id}_hits_counter'])}"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Distance covered
            y_offset += line_spacing
            text = "Distance Covered"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_distance_covered']:.1f} m"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Score probability
            y_offset += line_spacing
            text = "Score Probability"
            cv2.putText(stats_box_frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text = f"{row[f'player_{selected_player_id}_score_probability']:.1f}%"
            cv2.putText(stats_box_frame, text, (width//2, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the current frame to the video
        video_writer.write(stats_box_frame)

    # Release the video writer
    video_writer.release()

    return output_path


def map_court_position_to_player_id(refined_keypoints, player_detections):
    """
    Maps 'Upper' and 'Lower' court positions to player IDs (1 or 2).
    
    Args:
        refined_keypoints: Court keypoints
        player_detections: List or dictionary of player detections
    
    Returns:
        Dict mapping 'Upper' and 'Lower' to player IDs
    """
    # Calculate the net's y-coordinate (midpoint between keypoints 10 and 11)
    net_y = (refined_keypoints[10][1] + refined_keypoints[11][1]) / 2
    
    player_mapping = {}
    
    # Handling when player_detections is a list
    if isinstance(player_detections, list):
        # Find the first non-empty frame with detections
        for frame_idx, frame_data in enumerate(player_detections):
            if frame_data:  # If there are player detections in this frame
                for player_id, bbox in frame_data.items():
                    player_y = bbox[1] + bbox[3]/2
                    if player_y < net_y:
                        player_mapping['Upper'] = player_id
                    else:
                        player_mapping['Lower'] = player_id
                # Once we've processed one frame with detections, break
                if len(player_mapping) >= 2:
                    break
                
    # Handling when player_detections is a dictionary
    elif isinstance(player_detections, dict):
        sample_frame = next(iter(player_detections))
        
        for player_id in player_detections[sample_frame]:
            player_bbox = player_detections[sample_frame][player_id]
            player_y = player_bbox[1] + player_bbox[3]/2
            
            if player_y < net_y:
                player_mapping['Upper'] = player_id
            else:
                player_mapping['Lower'] = player_id
    
    return player_mapping