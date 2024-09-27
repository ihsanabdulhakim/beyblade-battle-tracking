# Global import
import numpy as np
import pandas as pd
import sqlite3
import time
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

# Load training model YOLOv8n
model = YOLO('./model/best.pt')

# Input video
video_input = './input/beybladebattle_input.mp4'
video_output = './output/beybladebattle_output.mp4'

# Transform function to convert PIL image to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
])

# Threshold for losing parameters
threshold_diff = 1.00           # Below this value is the boundaries for closest distance (assumption for object not moving at all)
mean_threshold = 0.20           # Below this value is the boundaries for similarity image (assumption for not changing looks for few frames)

# Initializing variables for the beyblade
beyblade1_id = beyblade2_id = None
beyblade1_array_before = beyblade2_array_before = None
beyblade1_pos = beyblade2_pos = None
beyblade1_pos_before = beyblade2_pos_before = None
beyblade1_conf = beyblade2_conf =None
beyblade1_cropped = beyblade2_cropped = None
beyblade1_win_count = beyblade2_win_count = 0
beyblade_detected_count = 0  # Counter for detected Beyblades

# Initializing time variables
frame_counter = 0
frame_battle_counter = 0
previous_timestamp = None  # To store the previous timestamp
winning1_enabled = winning2_enabled = False  # Statement to triggering winning condition

# Object ID tracker
track_history = {}
tracking_enabled = False # Statement to triggering object tracking
two_first_beyblades_frame = None # track first frame with two beyblades
last_known_beyblade_ids = {} # variable to saving last beyblades to handle frame missing problem
custom_id_map = {}  # Dictionary to map YOLO track ID to custom ID
next_custom_id = 1  # Start custom IDs from 1

# Create table (CSV) for analysis
df1_columns = [
    'Beyblade1_x', 'Beyblade1_y', 'Beyblade1_pos_diff', 'Beyblade1_array_diff', 'Beyblade1_conf', 
    'Beyblade2_x', 'Beyblade2_y', 'Beyblade2_pos_diff', 'Beyblade2_array_diff', 'Beyblade2_conf', 
    'Timeframe', 'Timestamp']
df1 = pd.DataFrame(columns=df1_columns)

# Create table (CSV) for battle results
df2_columns = ['Beyblade Win','Beyblade Lose','Duration']
df2 = pd.DataFrame(columns=df2_columns)


# Initialize text display variable
status_text = "Status : "
battlecountingtext = "Battle Duration: -"

# Function to create centroid coordinates
def center(x1,x2):
    centerresult = (x1 + x2) // 2
    return centerresult

# Function to store object (beyblade) features
def beyblade_array(frame,x1,x2,y1,y2,conf):
    center_x = center(x1,x2)
    center_y = center(y1,y2)
    beyblade_pos = (center_x, center_y)
    beyblade_conf = conf
    beyblade_cropped = frame[y1:y2, x1:x2]
    beyblade_convert = Image.fromarray(beyblade_cropped)  # Convert NumPy array to PIL image
    beyblade_array_this_fps = transform(beyblade_convert)
    beyblade_array_this_fps = beyblade_array_this_fps * 255.0
    beyblade_array_this_fps = beyblade_array_this_fps.type(torch.uint8)
    return beyblade_cropped, beyblade_array_this_fps, beyblade_pos, beyblade_conf

# Function to count duration
def timecounting(frame_counter,fps,text):
    timecounting = frame_counter / fps 
    minutes = int(timecounting // 60)  # Get total minutes
    seconds = int(timecounting % 60)    # Get remaining seconds

    # Format time in mm:ss
    timecountingtext = f"{text}: {minutes:02}:{seconds:02} s"
    return timecountingtext

# Function to store last data as new variables in next frame
def updatearray(beyblade1_array_this_fps,beyblade2_array_this_fps,beyblade1_pos,beyblade2_pos,timestamp):
    beyblade1_array_before = beyblade1_array_this_fps.clone()  # Store current frame image
    beyblade2_array_before = beyblade2_array_this_fps.clone()  # Store current frame image
    beyblade1_pos_before = beyblade1_pos  # Store current position
    beyblade2_pos_before = beyblade2_pos  # Store current position
    previous_timestamp = timestamp
    return beyblade1_array_before, beyblade2_array_before, beyblade1_pos_before, beyblade2_pos_before, previous_timestamp

# Function to calculate two-points distance
def euclidandistance(beyblade_pos,beyblade_pos_before):
    y = ((beyblade_pos[0] - beyblade_pos_before[0]) ** 2 + (beyblade_pos[1] - beyblade_pos_before[1]) ** 2) ** 0.5
    return round(y,2)

# Function to calculate average value of substraction between two arrays or tensor of image
def meanarraydifference(beyblade_array_this_fps,beyblade_array_before):
    y = beyblade_array_this_fps.float() - beyblade_array_before.float()
    return round((y).mean().item(),2)

# Function to print terminal statement of data stored in the table (CSV)
def printbeyblade(custom_track_id, beyblade_pos,beyblade_conf_value,beyblade_pos_diff,mean_diff_beyblade):
    print(f"Beyblade {custom_track_id} position: {beyblade_pos[0]},{beyblade_pos[1]}, conf: {beyblade_conf_value:.2f}, elimination position: {beyblade_pos_diff}, mean elimination array: {mean_diff_beyblade}")
    return

# Function to initialized beyblade color
def get_color_for_id(custom_track_id):
    color_map = {
        1: (0, 0, 255),    # Blue for beyblade 1
        2: (255, 0, 0),    # Red for beyblade 2
    }
    return color_map.get(custom_track_id, (0, 255, 0))  

# Function to creating video records of object (beyblades) tracking
def making_scatter_tracking(track_history, tracking_enabled, height, width, out_beyblade1, out_beyblade2):
    # Create white background
    tracking_frame_beyblade1 = np.full((height, width, 3), 255, dtype=np.uint8)  # White background
    tracking_frame_beyblade2 = np.full((height, width, 3), 255, dtype=np.uint8)  # White background

    # Draw the trail dot
    if tracking_enabled:
        for track_id, centers in track_history.items():
            track_color = get_color_for_id(track_id)  # Use the same color for tracking
            for center in centers:
                # Draw the tracking point on the appropriate tracking frame
                if track_id == 1:
                    # Draw a black circle for the border
                    cv2.circle(tracking_frame_beyblade1, center, 3, (0, 0, 0), -1)  # Border circle
                    # Draw the filled circle for the tracking point
                    cv2.circle(tracking_frame_beyblade1, center, 2, track_color, -1)  # Actual tracking point
                elif track_id == 2:
                    # Draw a black circle for the border
                    cv2.circle(tracking_frame_beyblade2, center, 3, (0, 0, 0), -1)  # Border circle
                    # Draw the filled circle for the tracking point
                    cv2.circle(tracking_frame_beyblade2, center, 2, track_color, -1)  # Actual tracking point

    # Write output file name
    out_beyblade1.write(tracking_frame_beyblade1)
    out_beyblade2.write(tracking_frame_beyblade2)

# Read video
cap = cv2.VideoCapture(video_input)
print(f"Read {video_input}")

# Initialize video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

# Video writer for tracking detection
beyblade1_tracking_output = './output/beyblade1-tracking-trails.mp4'
beyblade2_tracking_output = './output/beyblade2-tracking-trails.mp4'
out_beyblade1 = cv2.VideoWriter(beyblade1_tracking_output, fourcc, fps, (width, height))
out_beyblade2 = cv2.VideoWriter(beyblade2_tracking_output, fourcc, fps, (width, height))

# Loop the video
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame_counter += 1  # Add the frame counter

    if success:
        # Run model
        results = model.track(frame, persist=True, verbose=False)  # Disable verbose output

        # Track the number of unique beyblade ID
        unique_beyblade_ids = set()
        detected_beyblade_count = 0

        for result in results:
            for det in result.boxes:
                conf = det.conf[0].item()
                cls_id = int(det.cls[0].item())             # Class ID; '0' for beyblades, '1' for hand, and '2' for launcher
                x1, y1, x2, y2 = map(int, det.xyxy[0])      # Bounding box edge coordinates
                if conf > 0.7:                              # Confidence threshold
                    if cls_id == 1 or cls_id == 2:          # Condition for hand / launcher
                        status_text = "Status : Starting"   # Rewrite the status display
                        if cls_id == 1:
                            labels = 'hand'
                            colors = (128, 0, 128)
                        else:
                            labels = 'launcher'
                            colors = (192, 192, 192)
                        # Display bounding box and text
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colors, 1)
                        cv2.putText(frame, labels, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors, 2)

                    # Condition for beyblades
                    elif cls_id == 0:
                        detected_beyblade_count += 1 
                        beyblade_detected_count += 1

                        if det.id is None:
                            continue
                        track_id = int(det.id[0].item())

                        if track_id not in custom_id_map:
                            custom_id_map[track_id] = next_custom_id
                            next_custom_id += 1 
                        
                        # Get the custom ID
                        custom_track_id = custom_id_map[track_id]
                        unique_beyblade_ids.add(custom_track_id)

                        # Reassigned missing temporary frames
                        if next_custom_id > 2:
                            if beyblade1_cropped is None and beyblade2_cropped is not None:
                                custom_track_id = 1
                            elif beyblade2_cropped is None and beyblade1_cropped is not None:
                                custom_track_id = 2
                        
                        # Condition for 2 beyblades
                        if custom_track_id == 1:
                            beyblade1_cropped,beyblade1_array_this_fps,beyblade1_pos,beyblade1_conf = beyblade_array(frame,x1,x2,y1,y2,conf)
                            beyblade1_id = custom_track_id
                        elif custom_track_id == 2:
                            beyblade2_cropped,beyblade2_array_this_fps,beyblade2_pos,beyblade2_conf = beyblade_array(frame,x1,x2,y1,y2,conf)
                            beyblade2_id = custom_track_id

                        # Get the color for the bounding box
                        bbox_color = get_color_for_id(custom_track_id)
                        label = f"Beyblade-{custom_track_id}, conf: {conf:.2f}"
                        # Display the bounding box of beyblades
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

                        # Add the center point to the track history for the current object
                        if custom_track_id not in track_history:
                            track_history[custom_track_id] = []
                        track_history[custom_track_id].append((center(x1,x2),center(y1,y2)))
                else:
                    continue

        # Use the last known ID if the there are no current array detected
        if beyblade1_cropped is None:
            beyblade1_pos = last_known_beyblade_ids.get(beyblade1_id)
        if beyblade2_cropped is None:
            beyblade2_pos = last_known_beyblade_ids.get(beyblade2_id)

        # Save the last known positions
        if beyblade1_pos:
            last_known_beyblade_ids[beyblade1_id] = beyblade1_pos
        if beyblade2_pos:
            last_known_beyblade_ids[beyblade2_id] = beyblade2_pos

        # Enable tracking and visualization only if at least 2 beyblades are detected
        if beyblade1_pos and beyblade2_pos:
            tracking_enabled = True
            frame_battle_counter += 1

            # Capturing the first frame
            if two_first_beyblades_frame is None:
                two_first_beyblades_frame = frame_counter
            timestamp = (frame_counter - two_first_beyblades_frame) / fps
            battlecountingtext = timecounting(frame_battle_counter,fps,'Battle Duration: ')
            
            # Calculate the average difference for both beyblades tensor 
            if beyblade1_array_before is not None and beyblade2_array_before is not None:
                mean_diff_beyblade1 = meanarraydifference(beyblade1_array_this_fps,beyblade1_array_before)
                mean_diff_beyblade2 = meanarraydifference(beyblade2_array_this_fps,beyblade2_array_before)
            else:
                mean_diff_beyblade1 = 0.00
                mean_diff_beyblade2 = 0.00

            # Calculate the position difference for both beyblades
            if beyblade1_pos_before is not None and beyblade2_pos_before is not None:
                beyblade1_pos_diff = euclidandistance(beyblade1_pos,beyblade1_pos_before)
                beyblade2_pos_diff = euclidandistance(beyblade2_pos,beyblade2_pos_before)
            else:
                beyblade1_pos_diff = 0.00
                beyblade2_pos_diff = 0.00

            # Parameter for losing beyblade condition
            if beyblade1_pos_diff < threshold_diff:
                if np.abs(mean_diff_beyblade1) < mean_threshold:  # Check mean difference for beyblade 1
                    beyblade2_win_count += 1  # Increasing the win for the beyblade 2 (opponent)
            else:
                beyblade2_win_count = 0  # Reset counter if condition is not suit

            if beyblade2_pos_diff < threshold_diff:
                if np.abs(mean_diff_beyblade2) < mean_threshold:  # Check mean difference for beyblade 2
                    beyblade1_win_count += 1  # Increasing the win for the beyblade 1 (opponent)
            else:
                beyblade1_win_count = 0  # Reset counter if condition is not suit

            # Store the positions of the two Beyblades, the timeframe, and the timestamp
            timeframe = frame_counter / fps  # Calculate the current time

            # Store the data into table (CSV)
            df1_columns_value = [
                beyblade1_pos[0], beyblade1_pos[1], mean_diff_beyblade1, beyblade1_pos_diff, beyblade1_conf,
                beyblade2_pos[0], beyblade2_pos[1], mean_diff_beyblade2, beyblade2_pos_diff, beyblade2_conf,
                timeframe, timestamp]
            df1.loc[len(df1)] = df1_columns_value
            df1.to_csv('./output/beyblade-tracking-analyzed.csv', index=False) 

            # Assuming beyblade1_conf and beyblade2_conf can be None
            beyblade1_conf_value = beyblade1_conf if beyblade1_conf is not None else 0.0
            beyblade2_conf_value = beyblade2_conf if beyblade2_conf is not None else 0.0

            # Print the positions, position differences, and timestamp to the console
            print("")
            if previous_timestamp is None:
                differencetimestamp = 0
            else:
                differencetimestamp = round(timestamp,2) - round(previous_timestamp,2)
            print(f"timeframe: {timeframe:.2f} s, timestamp: {timestamp:.2f} s, Difference timestamp: {differencetimestamp} s") 
            printbeyblade(1,beyblade1_pos,beyblade1_conf_value,beyblade1_pos_diff,mean_diff_beyblade1)
            printbeyblade(2,beyblade2_pos,beyblade2_conf_value,beyblade2_pos_diff,mean_diff_beyblade2)
            

            if beyblade1_cropped is not None and beyblade2_cropped is not None:
                print("")
                if beyblade1_win_count >= 3:
                    print("Beyblade 1 Wins the Match!")
                    winning1_enabled = True
                    cv2.imwrite('./output/beybladewin-1.jpg', beyblade1_cropped)
                    cv2.imwrite('./output/beybladelose-2.jpg', beyblade2_cropped)
                    df2_columns_value = [int(1), int(2), round(timestamp,2)]              # Integrate data result to table (CSV)
                    df2.loc[len(df2)] = df2_columns_value                       
                    beyblade1_win_count = 0 
                                                                                # Reset count after declaring a winner
                elif beyblade2_win_count >= 3:
                    print("Beyblade 2 Wins the Match!")
                    winning2_enabled = True
                    cv2.imwrite('./output/beybladewin-2.jpg', beyblade2_cropped)
                    cv2.imwrite('./output/beybladelose-1.jpg', beyblade1_cropped)
                    df2_columns_value = [int(2), int(1), round(timestamp,2)]
                    df2.loc[len(df2)] = df2_columns_value
                    beyblade2_win_count = 0  

            if beyblade1_pos and beyblade2_pos:
                beyblade1_array_before, beyblade2_array_before, beyblade1_pos_before, beyblade2_pos_before, previous_timestamp = updatearray(beyblade1_array_this_fps,beyblade2_array_this_fps,beyblade1_pos,beyblade2_pos,timestamp)

        # Update status to "Battle" if the second Beyblade is detected
        if tracking_enabled:
            if beyblade_detected_count >= 2:
                status_text = "Status : Battle"     
        if winning1_enabled:
            status_text = "Status : End, Beyblade 1 is the winner!"     # Update status display
        elif winning2_enabled:
            status_text = "Status : End, Beyblade 2 is the winner!"     

        # Calculate elapsed time
        timecountingtext = timecounting(frame_counter,fps,'Video Time: ')
        cv2.putText(frame, timecountingtext, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, battlecountingtext, (550, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Write the frame to the output video (with bounding boxes and status text)
        out.write(frame)

        # Display the annotated frame
        cv2.imshow("Beyblade Battle Tracker - YOLOv8n", frame)

        # Making video for beyblade tracking trail
        making_scatter_tracking(track_history, tracking_enabled, height, width, out_beyblade1, out_beyblade2)

        # After exiting the loop, release resources and output the duration
        if winning1_enabled or winning2_enabled:
            print(f"Total duration of the battle {timestamp:.2f} s")
            print("")
            df2.to_csv('./output/beyblade_battle_result.csv', index=False)

            # Save as database SQLite
            db_path = './output/beybladebattletracking.db'
            conn = sqlite3.connect(db_path)             # connect to sqlite
            df1.to_sql('beyblade_tracking', conn, if_exists='replace', index=False)
            df2.to_sql('beyblade_battle_result', conn, if_exists='replace', index=False)


            time.sleep(5) # Give 5 seconds before closing the program
            
            # Release resources
            cap.release()
            out.release()
            conn.commit()
            conn.close()
            out_beyblade1.release()
            out_beyblade2.release()
            cv2.destroyAllWindows()
           
            # You can exit here
        else:
            # Continue processing until the video ends
            pass

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Clean up resources
cap.release()
out.release()
out_beyblade1.release()
out_beyblade2.release()
cv2.destroyAllWindows()







    
