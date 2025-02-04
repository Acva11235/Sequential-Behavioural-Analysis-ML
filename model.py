import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import pynput
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener
import threading
import queue
import sys
import math

class UserBehaviorAnalyzer:
    def __init__(self):
        # Existing variables
        self.keyboard_queue = queue.Queue()
        self.mouse_queue = queue.Queue()
        self.keyboard_events = []
        self.mouse_events = []
        self.key_timestamps = []
        self.key_presses = 0
        self.key_intervals = []
        self.mouse_clicks = []
        self.mouse_positions = []
        self.mouse_click_timestamps = []
        self.stop_event = threading.Event()
        self.keyboard_listener = None
        self.mouse_listener = None

        # New tracking variables
        self.mouse_speeds = []

    def reset_listeners(self):
        # Existing resets
        self.keyboard_queue = queue.Queue()
        self.mouse_queue = queue.Queue()
        self.keyboard_events = []
        self.mouse_events = []
        self.key_timestamps = []
        self.key_presses = 0
        self.key_intervals = []
        self.mouse_clicks = []
        self.mouse_positions = []
        self.mouse_click_timestamps = []
        self.stop_event.clear()
        
        # New resets
        self.mouse_speeds = []

        # Listeners remain same
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
        # sys.stdout.write(f"Key Press: {event}\n")
        # sys.stdout.flush()

    def on_key_release(self, key):
        current_time = time.time()
        event = {
            'type': 'key_release',
            'key': str(key),
            'timestamp': current_time
        }
        self.keyboard_queue.put(event)
        # sys.stdout.write(f"Key Release: {event}\n")
        # sys.stdout.flush()

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
        # sys.stdout.write(f"Mouse Click: {event}\n")
        # sys.stdout.flush()

    def on_mouse_move(self, x, y):
        current_time = time.time()
        event = {
            'type': 'mouse_move',
            'x': x,
            'y': y,
            'timestamp': current_time
        }
        self.mouse_queue.put(event)
        # sys.stdout.write(f"Mouse Move: {event}\n")
        # sys.stdout.flush()

    def event_processor(self):
        """Process events from queues and update feature tracking"""
        while not self.stop_event.is_set():
            # Existing keyboard processing
            while not self.keyboard_queue.empty():
                event = self.keyboard_queue.get()
                self.keyboard_events.append(event)
                
                if event['type'] == 'key_press':
                    current_time = event['timestamp']
                    self.key_presses += 1
                    
                    if self.key_timestamps:
                        interval = current_time - self.key_timestamps[-1]
                        self.key_intervals.append(interval)
                    
                    self.key_timestamps.append(current_time)

            # Modified mouse processing
            while not self.mouse_queue.empty():
                event = self.mouse_queue.get()
                self.mouse_events.append(event)
                
                # Existing click handling
                if event['type'] == 'mouse_click':
                    self.mouse_clicks.append((event['x'], event['y']))
                    self.mouse_click_timestamps.append(event['timestamp'])
                
                # New mouse speed tracking
                if event['type'] == 'mouse_move':
                    self.mouse_positions.append((event['x'], event['timestamp']))
                    if len(self.mouse_positions) > 1:
                        prev_x, prev_time = self.mouse_positions[-2]
                        prev_y, curr_time = self.mouse_positions[-1]
                        dx = event['x'] - prev_x
                        dy = event['y'] - prev_y
                        distance = math.sqrt(dx**2 + dy**2)
                        time_diff = curr_time - prev_time
                        if time_diff > 0:
                            self.mouse_speeds.append(distance / time_diff)

            time.sleep(0.01)

    def extract_features(self, duration):
        # 1. Typing speed (cpm)
        typing_speed = (self.key_presses / duration) * 60 if duration > 0 else 0
        
        # 2. Shortcut keys pressed
        active_modifiers = set()
        shortcut_count = 0
        modifiers = {'Key.ctrl', 'Key.ctrl_l', 'Key.ctrl_r', 'Key.alt', 'Key.alt_l', 
                    'Key.alt_r', 'Key.cmd', 'Key.shift', 'Key.shift_l', 'Key.shift_r'}
        for event in self.keyboard_events:
            key = event['key']
            if event['type'] == 'key_press':
                if key in modifiers:
                    active_modifiers.add(key)
                else:
                    if active_modifiers:
                        shortcut_count += 1
            elif event['type'] == 'key_release' and key in modifiers:
                if key in active_modifiers:
                    active_modifiers.discard(key)
        
        # 3. Correction count (backspace/delete)
        correction_count = sum(
            1 for e in self.keyboard_events 
            if e['type'] == 'key_press' and e['key'] in {'Key.backspace', 'Key.delete'}
        )
        
        # 4. Dwell time calculation
        pressed_keys = {}
        dwell_times = []
        for event in self.keyboard_events:
            if event['type'] == 'key_press':
                pressed_keys[event['key']] = event['timestamp']
            elif event['type'] == 'key_release':
                if event['key'] in pressed_keys:
                    dwell = event['timestamp'] - pressed_keys.pop(event['key'])
                    dwell_times.append(dwell)
        avg_dwell = sum(dwell_times)/len(dwell_times) if dwell_times else 0
        
        # 5. Flight time calculation
        flight_times = []
        last_release = None
        for event in self.keyboard_events:
            if event['type'] == 'key_release':
                last_release = event['timestamp']
            elif event['type'] == 'key_press' and last_release is not None:
                flight_times.append(event['timestamp'] - last_release)
                last_release = None
        avg_flight = sum(flight_times)/len(flight_times) if flight_times else 0
        
        # 6. Mouse-key interval calculation
        import bisect
        mouse_times = [e['timestamp'] for e in self.mouse_events]
        key_times = [e['timestamp'] for e in self.keyboard_events if e['type'] == 'key_press']
        sorted_mouse = sorted(mouse_times)
        intervals = []
        for kt in key_times:
            idx = bisect.bisect_left(sorted_mouse, kt)
            if idx == 0 and sorted_mouse:
                diff = sorted_mouse[0] - kt
            elif idx >= len(sorted_mouse):
                diff = kt - sorted_mouse[-1] if sorted_mouse else 0
            else:
                before = sorted_mouse[idx-1]
                after = sorted_mouse[idx]
                diff = min(kt - before, after - kt)
            intervals.append(diff)
        avg_mouse_interval = sum(intervals)/len(intervals) if intervals else 0
        
        # 7. Mouse click distance
        click_distances = []
        for i in range(1, len(self.mouse_clicks)):
            x1, y1 = self.mouse_clicks[i-1]
            x2, y2 = self.mouse_clicks[i]
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            click_distances.append(distance)
        avg_click_distance = sum(click_distances)/len(click_distances) if click_distances else 0

        # New Feature 1: Average Mouse Speed
        avg_mouse_speed = np.mean(self.mouse_speeds) if self.mouse_speeds else 0

        # New Feature 2: Double Click Count
        double_clicks = 0
        click_times = self.mouse_click_timestamps
        for i in range(1, len(click_times)):
            if click_times[i] - click_times[i-1] < 0.5:  # 500ms threshold
                double_clicks += 1

        # Update print statements
        print("\n--- Extracted Features ---")
        # Existing prints
        print(f"Typing Speed (keys/min): {typing_speed:.2f}")
        print(f"Shortcut Keys Pressed: {shortcut_count}")
        print(f"Corrections (Backspace/Delete): {correction_count}")
        print(f"Average Dwell Time (s): {avg_dwell:.3f}")
        print(f"Average Flight Time (s): {avg_flight:.3f}")
        print(f"Average Mouse-Key Interval (s): {avg_mouse_interval:.3f}")
        print(f"Average Click Distance (px): {avg_click_distance:.1f}")
        
        # New prints
        print(f"Average Mouse Speed (px/s): {avg_mouse_speed:.1f}")
        print(f"Double Click Count: {double_clicks}")

        return [
            # Existing features
            typing_speed,
            shortcut_count,
            correction_count,
            avg_dwell,
            avg_flight,
            avg_mouse_interval,
            avg_click_distance,
            
            # New features added at end
            avg_mouse_speed,
            double_clicks
        ]

    def collect_data(self, duration=120):
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
        # Collect multiple training samples
        training_features = []
        for _ in range(5):  # Collect 5 training samples
            self.collect_data(duration=20)
            features = self.extract_features(duration=20)
            training_features.append(features)
        
        # Convert to numpy array for standardization
        X_train = np.array(training_features)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Isolation Forest with higher contamination
        clf = IsolationForest(
            contamination=0.3,  # Increased contamination factor
            random_state=42, 
            max_samples='auto',
            bootstrap=True,
            n_estimators=100  # Increased number of trees
        )
        clf.fit(X_train_scaled)
        
        # Run inference
        print("\n--- Inference Results ----------------------------------------------------------------------------------------------------------")
        for i in range(5):
            # Collect data for 10 seconds
            self.collect_data(duration=10)
            
            # Extract features
            inference_features = self.extract_features(duration=10)
            
            # Scale inference features using the same scaler
            inference_scaled = scaler.transform([inference_features])
            
            # Predict
            prediction = clf.predict(inference_scaled)
            
            print(f"Inference {i+1}: {'Normal' if prediction[0] == 1 else 'Anomaly Detected'}")
            print("Features:", inference_features)
            print("---"*15)

def main():
    analyzer = UserBehaviorAnalyzer()
    analyzer.run_intrusion_detection()

if __name__ == "__main__":
    main()