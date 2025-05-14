from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import numpy as np
import os
from model import detect_vehicles

app = Flask(__name__)

# Configure video paths
video_paths = [
    "Dynamic-Traffic-light-prediction-project/src/data/A.mp4",
    "Dynamic-Traffic-light-prediction-project/src/data/C.mp4",
    "Dynamic-Traffic-light-prediction-project/src/data/D.mp4",
    "Dynamic-Traffic-light-prediction-project/src/data/B.mp4"
]

# Configuration parameters
base_time, max_time = 10, 180
yellow_duration = 3  # Yellow signal duration

# Shared state
lane_data = [{"frame": None, "count": 0, "last_update": 0} for _ in range(4)]
green_index = 0
time_remaining = 0
signal_states = ["RED", "RED", "RED", "RED"]
signal_states[0] = "GREEN"  # Initial state
red_timers = [0, 0, 0, 0]  # Initialize red timers for each lane

# Threading locks
state_lock = threading.Lock()
video_locks = [threading.Lock() for _ in range(4)]

# Create frames dictionary to store the latest frames
frames = [{
    "original": None,
    "processed": None,
    "count": 0
} for _ in range(4)]

def capture_thread(lane_index):
    """Thread to continuously capture frames from each video source"""
    cap = cv2.VideoCapture(video_paths[lane_index])
    
    if not cap.isOpened():
        print(f"Error: Could not open video capture for lane {lane_index+1}")
        return
        
    print(f"Started capture thread for lane {lane_index+1}")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"Resetting video for lane {lane_index+1}")
                cap.release()
                time.sleep(0.5) 
                cap = cv2.VideoCapture(video_paths[lane_index])
                if not cap.isOpened():
                    print(f"Failed to reopen video for lane {lane_index+1}")
                    time.sleep(1)
                    continue
            else:
                # Store the original frame
                with video_locks[lane_index]:
                    # Resize for consistency
                    frame = cv2.resize(frame, (640, 480))
                    frames[lane_index]["original"] = frame.copy()
                    
                    # If no processed frame exists yet, use the original
                    if frames[lane_index]["processed"] is None:
                        frames[lane_index]["processed"] = frame.copy()
                        
        except Exception as e:
            print(f"Error in capture thread for lane {lane_index+1}: {e}")
            time.sleep(0.5)
            
        # Short sleep to reduce CPU usage while maintaining smooth playback
        time.sleep(0.03)  # ~33 fps

def detect_thread():
    """Thread to perform vehicle detection when traffic light changes"""
    global green_index, time_remaining
    
    while True:
        # Wait for green light to change or initial detection
        time.sleep(1)
        
        with state_lock:
            current_green = green_index
            should_detect = time_remaining <= 1  # Detect when light is about to change
        
        if should_detect:
            # Run detection on all lanes
            for lane_index in range(4):
                with video_locks[lane_index]:
                    if frames[lane_index]["original"] is not None:
                        frame = frames[lane_index]["original"].copy()
                        
                # Process the frame with vehicle detection
                processed_frame, vehicle_count = detect_vehicles(frame)
                
                # Update the processed frame and count
                with video_locks[lane_index]:
                    frames[lane_index]["processed"] = processed_frame
                    frames[lane_index]["count"] = vehicle_count
                    
            print(f"Updated counts: {[frames[i]['count'] for i in range(4)]}")

def scheduler():
    """Traffic light scheduler thread"""
    global green_index, time_remaining, signal_states, red_timers
    
    # Initial detection to get started
    time.sleep(3)  # Wait for capture threads to get frames
    
    # Main scheduling loop
    while True:
        # Calculate green times based on current vehicle counts
        counts = [frames[i]['count'] for i in range(4)]
        total = sum(counts)
        
        if total > 0:
            green_times = [
                int(base_time + ((counts[i] / total) * (max_time - base_time)))
                for i in range(4)
            ]
        else:
            green_times = [base_time for _ in range(4)]
        
        # Set time for current green lane
        with state_lock:
            time_remaining = green_times[green_index]
            
            # Update red timers for all other lanes
            # Each red lane's timer will show how long before it will turn green
            total_time_before_green = time_remaining + yellow_duration
            for i in range(4):
                if i == green_index:
                    red_timers[i] = 0  # Green lane has no red timer
                else:
                    # Calculate position relative to current green lane
                    position = (i - green_index) % 4
                    
                    # Calculate time until this lane gets green
                    # Current green lane time + yellow + any lanes that come before this one
                    red_timers[i] = total_time_before_green + sum(green_times[(green_index + j) % 4] + yellow_duration 
                                                               for j in range(1, position))
            
            print(f"Lane {green_index+1} gets {time_remaining}s of green time. Counts: {counts}")
            print(f"Red timers: {red_timers}")
        
        # Wait for the green duration
        while time_remaining > 0:
            time.sleep(1)
            with state_lock:
                time_remaining -= 1
                # Update red timers every second
                for i in range(4):
                    if i != green_index and red_timers[i] > 0:
                        red_timers[i] -= 1
        
        # Yellow light phase
        with state_lock:
            signal_states[green_index] = "YELLOW"
            
        # Wait for yellow duration and update red timers during this time
        for _ in range(yellow_duration):
            time.sleep(1)
            with state_lock:
                for i in range(4):
                    if i != green_index and red_timers[i] > 0:
                        red_timers[i] -= 1
        
        # Change to the next lane
        with state_lock:
            signal_states[green_index] = "RED"
            green_index = (green_index + 1) % 4
            signal_states[green_index] = "GREEN"

def generate_frame(lane_index):
    """Generate frame for web display"""
    global green_index, time_remaining, signal_states, red_timers
    
    with video_locks[lane_index]:
        if frames[lane_index]["processed"] is not None:
            frame = frames[lane_index]["processed"].copy()
            count = frames[lane_index]["count"]
        elif frames[lane_index]["original"] is not None:
            frame = frames[lane_index]["original"].copy()
            frame = cv2.resize(frame, (640, 480))
            count = 0
        else:
            # Create empty frame if no frame is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            count = 0
    
    
    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    
    return frame_bytes

def generate_feed(lane_index):
    """Generator function for the video streaming route"""
    while True:
        frame_bytes = generate_frame(lane_index)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 fps

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    return Response(generate_feed(lane_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signal_status')
def signal_status():
    """API endpoint to get current signal status for all lanes"""
    with state_lock:
        status_data = {
            "signals": signal_states.copy(),
            "remaining": time_remaining,
            "red_timers": red_timers.copy(),
            "counts": [frames[i]['count'] for i in range(4)]
        }
    return jsonify(status_data)

if __name__ == '__main__':
    # Start capture threads for each lane
    for i in range(4):
        threading.Thread(target=capture_thread, args=(i,), daemon=True).start()
    
    # Start detection thread
    threading.Thread(target=detect_thread, daemon=True).start()
    
    # Start scheduler thread
    threading.Thread(target=scheduler, daemon=True).start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)



































































































































































































































