import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import pynput
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener
import threading
import queue
import sys

class UserBehaviorAnalyzer:
    def __init__(self):
        # Thread-safe queue for events
        self.keyboard_queue = queue.Queue()
        self.mouse_queue = queue.Queue()
        
        # Event collection containers
        self.keyboard_events = []
        self.mouse_events = []
        
        # Tracking variables
        self.key_timestamps = []
        self.key_intervals = []
        self.mouse_clicks = []
        self.mouse_positions = []
        self.mouse_click_timestamps = []
        
        # Synchronization events
        self.stop_event = threading.Event()
        
        # Listeners
        self.keyboard_listener = None
        self.mouse_listener = None

    def reset_listeners(self):
        # Reset queues
        self.keyboard_queue = queue.Queue()
        self.mouse_queue = queue.Queue()
        
        # Reset events and tracking variables
        self.keyboard_events = []
        self.mouse_events = []
        self.key_timestamps = []
        self.key_intervals = []
        self.mouse_clicks = []
        self.mouse_positions = []
        self.mouse_click_timestamps = []
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create new listeners
        self.keyboard_listener = KeyboardListener(
            on_press=self.on_key_press, 
            on_release=self.on_key_release
        )
        self.mouse_listener = MouseListener(
            on_click=self.on_mouse_click, 
            on_move=self.on_mouse_move
        )

    def on_key_press(self, key):
        current_time = time.time()
        event = {
            'type': 'key_press',
            'key': str(key),
            'timestamp': current_time
        }
        self.keyboard_queue.put(event)
        sys.stdout.write(f"Key Press: {event}\n")
        sys.stdout.flush()

    def on_key_release(self, key):
        current_time = time.time()
        event = {
            'type': 'key_release',
            'key': str(key),
            'timestamp': current_time
        }
        self.keyboard_queue.put(event)
        sys.stdout.write(f"Key Release: {event}\n")
        sys.stdout.flush()

    def on_mouse_click(self, x, y, button, pressed):
        current_time = time.time()
        event = {
            'type': 'mouse_click',
            'x': x,
            'y': y,
            'button': str(button),
            'pressed': pressed,
            'timestamp': current_time
        }
        self.mouse_queue.put(event)
        sys.stdout.write(f"Mouse Click: {event}\n")
        sys.stdout.flush()

    def on_mouse_move(self, x, y):
        current_time = time.time()
        event = {
            'type': 'mouse_move',
            'x': x,
            'y': y,
            'timestamp': current_time
        }
        self.mouse_queue.put(event)
        sys.stdout.write(f"Mouse Move: {event}\n")
        sys.stdout.flush()

    def event_processor(self):
        """Process events from queues and update feature tracking"""
        while not self.stop_event.is_set():
            # Process keyboard events
            while not self.keyboard_queue.empty():
                event = self.keyboard_queue.get()
                self.keyboard_events.append(event)
                
                if event['type'] == 'key_press':
                    current_time = event['timestamp']
                    if self.key_timestamps:
                        interval = current_time - self.key_timestamps[-1]
                        self.key_intervals.append(interval)
                    self.key_timestamps.append(current_time)

            # Process mouse events
            while not self.mouse_queue.empty():
                event = self.mouse_queue.get()
                self.mouse_events.append(event)
                
                if event['type'] == 'mouse_click':
                    self.mouse_clicks.append((event['x'], event['y']))
                    self.mouse_click_timestamps.append(event['timestamp'])
                
                if event['type'] == 'mouse_move':
                    self.mouse_positions.append((event['x'], event['y']))

            time.sleep(0.01)  # Prevent tight loop

    def extract_features(self, duration=120):
        # Keyboard features
        if len(self.key_timestamps) > 1:
            typing_speed = len(self.key_timestamps) / duration
            avg_key_interval = np.mean(self.key_intervals) if self.key_intervals else 0
        else:
            typing_speed = 0
            avg_key_interval = 0

        # Mouse features
        if len(self.mouse_clicks) > 1:
            # Calculate click distances
            click_distances = [np.sqrt((self.mouse_clicks[i][0] - self.mouse_clicks[i-1][0])**2 + 
                                        (self.mouse_clicks[i][1] - self.mouse_clicks[i-1][1])**2) 
                               for i in range(1, len(self.mouse_clicks))]
            
            # Calculate click intervals
            click_intervals = [self.mouse_click_timestamps[i] - self.mouse_click_timestamps[i-1] 
                               for i in range(1, len(self.mouse_click_timestamps))]
            
            avg_click_distance = np.mean(click_distances) if click_distances else 0
            avg_click_interval = np.mean(click_intervals) if click_intervals else 0
        else:
            avg_click_distance = 0
            avg_click_interval = 0

        # Features for model
        features = [
            typing_speed,
            avg_key_interval,
            avg_click_distance,
            avg_click_interval
        ]

        print("\n--- Extracted Features ---")
        print(f"Typing Speed: {typing_speed}")
        print(f"Average Key Interval: {avg_key_interval}")
        print(f"Average Click Distance: {avg_click_distance}")
        print(f"Average Click Interval: {avg_click_interval}")

        return features

    def collect_data(self, duration=20):
        # Reset listeners for each data collection
        self.reset_listeners()
        
        # Start listeners
        self.keyboard_listener.start()
        self.mouse_listener.start()
        
        # Start event processor thread
        processor_thread = threading.Thread(target=self.event_processor)
        processor_thread.start()
        
        print(f"Starting data collection for {duration} seconds...")
        start_time = time.time()
        
        # Collect data for specified duration
        time.sleep(duration)
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Stop listeners
        self.keyboard_listener.stop()
        self.mouse_listener.stop()
        processor_thread.join()
        
        print("Data collection complete.")

    def run_intrusion_detection(self):
        # Collect training data
        self.collect_data(duration=20)
        
        # Prepare training data
        training_features = []
        features = self.extract_features(duration=20)
        training_features.append(features)
        
        # Train Isolation Forest
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(training_features)
        
        # Run inference
        print("\n--- Inference Results ---")
        for i in range(5):
            # Collect data for 10 seconds
            self.collect_data(duration=10)
            
            # Extract features
            inference_features = self.extract_features(duration=10)
            
            # Predict
            prediction = clf.predict([inference_features])
            
            print(f"Inference {i+1}: {'Normal' if prediction[0] == 1 else 'Anomaly Detected'}")
            print("Features:", inference_features)
            print("---")

def main():
    analyzer = UserBehaviorAnalyzer()
    analyzer.run_intrusion_detection()

if __name__ == "__main__":
    main()