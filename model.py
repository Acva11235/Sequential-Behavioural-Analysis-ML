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
import win32gui
import win32process
import psutil
import joblib
# from sklearn.externals import joblib


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
        # Nimish 
        self.app_queue = queue.Queue()
        self.open_apps = {}  # {hwnd: {title, exe, start, end}}
        self.focus_history = []  # List of focused apps with timestamps
        self.app_sessions = []  # Closed app sessions
        self.app_snapshots = []  # Periodic app state snapshots
        self.current_focus = None
        self.anomaly_count = 0

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
        # Nimish 
        self.app_queue = queue.Queue()
        self.open_apps = {}  # {hwnd: {title, exe, start, end}}
        self.focus_history = []  # List of focused apps with timestamps
        self.app_sessions = []  # Closed app sessions
        self.app_snapshots = []  # Periodic app state snapshots
        self.current_focus = None

        # Listeners remain same
        self.keyboard_listener = KeyboardListener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.mouse_listener = MouseListener(
            on_click=self.on_mouse_click,
            on_move=self.on_mouse_move
        )


    def _get_exe_path(self, hwnd):
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return psutil.Process(pid).exe()
        except:
            return "unknown"
        
    def app_monitor(self):
        """Thread to monitor application states"""
        last_scan = 0
        while not self.stop_event.is_set():
            try:
                # Track focused window
                current_time = time.time()
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                exe = self._get_exe_path(hwnd)
                
                if hwnd != self.current_focus:
                    self.app_queue.put({
                        'type': 'focus_change',
                        'hwnd': hwnd,
                        'title': title,
                        'exe': exe,
                        'timestamp': current_time
                    })
                
                # Full scan every 10 seconds
                if current_time - last_scan > 30:
                    last_scan = current_time
                    windows = []
                    def enum_callback(hwnd, _):
                        if win32gui.IsWindowVisible(hwnd):
                            title = win32gui.GetWindowText(hwnd)
                            windows.append((hwnd, title))
                    win32gui.EnumWindows(enum_callback, None)
                    
                    self.app_queue.put({
                        'type': 'full_scan',
                        'windows': windows,
                        'timestamp': current_time
                    })
                
                time.sleep(0.5)
            except Exception as e:
                print(f"App monitoring error: {str(e)}")


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

            # Process application events
            while not self.app_queue.empty():
                event = self.app_queue.get()
                
                if event['type'] == 'focus_change':
                    # Update focus history
                    if self.focus_history and not self.focus_history[-1]['end']:
                        self.focus_history[-1]['end'] = event['timestamp']
                    
                    self.focus_history.append({
                        'hwnd': event['hwnd'],
                        'title': event['title'],
                        'exe': event['exe'],
                        'start': event['timestamp'],
                        'end': None
                    })
                    self.current_focus = event['hwnd']
                
                elif event['type'] == 'full_scan':
                    # Update open apps
                    current_hwnds = {hwnd for hwnd, _ in event['windows']}
                    
                    # Detect closed apps
                    for hwnd in list(self.open_apps.keys()):
                        if hwnd not in current_hwnds:
                            closed_app = self.open_apps.pop(hwnd)
                            closed_app['end'] = event['timestamp']
                            self.app_sessions.append(closed_app)
                    
                    # Detect new apps
                    for hwnd, title in event['windows']:
                        if hwnd not in self.open_apps:
                            self.open_apps[hwnd] = {
                                'title': title,
                                'exe': self._get_exe_path(hwnd),
                                'start': event['timestamp'],
                                'end': None
                            }
                    
                    # Save snapshot
                    self.app_snapshots.append({
                        'timestamp': event['timestamp'],
                        'open_apps': list(self.open_apps.values())
                    })


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

        # 1. Application session statistics
        session_durations = [s['end'] - s['start'] for s in self.app_sessions if s['end']]
        len_session_durations =len(session_durations)
        
        
        # 2. Focus behavior
        focus_changes = len([f for f in self.focus_history if f['end']])
        focus_durations = [f['end'] - f['start'] for f in self.focus_history if f['end']]
        
        focus_rate = focus_changes / duration if duration > 0 else 0
        mean_focus_durations = np.mean(focus_durations) if focus_durations else 0
        

        #3 Tranistions
        transitions = []
        prev_exe = None
        for f in self.focus_history:
            if prev_exe and f['exe'] != prev_exe:
                transitions.append(f"{prev_exe}→{f['exe']}")
            prev_exe = f['exe']
        count_transitions =len(transitions)
        num_transition= len(set(transitions))
        rate_transition = len(transitions) / duration if duration > 0 else 0
        
        
        # 4. Concurrent apps
        concurrency = [len(s['open_apps']) for s in self.app_snapshots]
        mean_concurrency =np.mean(concurrency) if concurrency else 0
        
        

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

        print("\n--- Extracted Application Usage Features ---")
        
        print(f"Count of Completed Sessions: {len_session_durations}")
        print(f"Focus Rate (changes/s): {focus_rate:.2f}")
        print(f"Mean Focus Durations (s): {mean_focus_durations:.2f}")
        print(f"Transition Count: {count_transitions}")
        print(f"Unique Transitions: {num_transition}")
        print(f"Transition Rate (transitions/s): {rate_transition:.2f}")

        print(f"Mean Concurrency: {mean_concurrency:.2f}")
        

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
            double_clicks,

            
            len_session_durations,
            focus_rate*100,
            mean_focus_durations,
            count_transitions,
            num_transition,
            rate_transition*100,
            mean_concurrency,
           

        ]

    def collect_data(self, duration=30):
        # Reset listeners for each data collection
        self.reset_listeners()
        
        # Start listeners
        self.keyboard_listener.start()
        self.mouse_listener.start()
        
        # Start event processor thread
        processor_thread = threading.Thread(target=self.event_processor)
        processor_thread.start()

        app_monitor_thread = threading.Thread(target=self.app_monitor)
        app_monitor_thread.start()
        
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
        app_monitor_thread.join()
        
        print("Data collection complete.")

    def run_intrusion_detection(self, model_filename="intrusion_model.joblib"):
        # Collect multiple training samples
        training_features = []
        for _ in range(10):  # or however many samples you want
            self.collect_data(duration=60)
            features = self.extract_features(duration=30)
            training_features.append(features)

        # Convert to numpy array for standardization
        X_train = np.array(training_features)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Isolation Forest
        self.clf = IsolationForest(
            contamination=0.38,
            random_state=42,
            max_samples='auto',
            bootstrap=True,
            n_estimators=200
        )
        self.clf.fit(X_train_scaled)

        # Save both the trained model and the scaler
        joblib.dump(self.clf, model_filename)
        joblib.dump(self.scaler, model_filename.replace(".joblib", "_scaler.joblib"))
        print(f"Model saved to {model_filename}")
        print(f"Scaler saved to {model_filename.replace('.joblib', '_scaler.joblib')}")

  
    def load_model(self, model_filename="intrusion_model.joblib"):
        try:
            self.clf = joblib.load(model_filename)  # Load the IsolationForest
            self.scaler = joblib.load(model_filename.replace(".joblib", "_scaler.joblib"))  # Load the scaler
            print(f"Model loaded from {model_filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {model_filename} not found. Training a new model.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def run_inference(self, model_filename="intrusion_model.joblib"):
    # Load model (and scaler)
        if not self.load_model(model_filename):
            # If loading fails, it trains a new model and stops
            # Alternatively, you could just call run_intrusion_detection here
            self.run_intrusion_detection(model_filename)
            return

        # Now run inference
        print("\n--- Inference Results --------------------------------------------------------------------")
        for i in range(7):
            self.collect_data(duration=30)
            inference_features = self.extract_features(duration=30)
            
            # Scale with the same scaler
            inference_scaled = self.scaler.transform([inference_features])
            
            # Predict
            prediction = self.clf.predict(inference_scaled)
            is_anomaly = prediction[0] == -1
            if is_anomaly:
                self.anomaly_count += 1
                print(f"! ALERT: Anomalies detected ({self.anomaly_count}/2) !")
                
                # Check if threshold reached
                if self.anomaly_count >= 2:
                    print("\n! SYSTEM FREEZE ! Multiple anomalies detected.")
                    time.sleep(10)
                    print("Resuming operations...")
                    self.anomaly_count = 0  

            print(f"Inference {i+1}: {'Normal' if prediction[0] == 1 else 'Anomaly Detected'}")
            print("Features:", inference_features)
            print("---" * 15)


def main():
    analyzer = UserBehaviorAnalyzer()
    # Check if the model exists, if not train and save it
    if not analyzer.load_model():
        analyzer.run_intrusion_detection()

    # Run Inference (you can call this separately later without retraining)
    
    analyzer.run_inference()


if __name__ == "__main__":
    main()