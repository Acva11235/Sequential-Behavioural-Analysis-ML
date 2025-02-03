import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import pynput
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener
import threading
import queue

class UserBehaviorAnalyzer:
    def __init__(self):
        self.keyboard_queue = queue.Queue()
        self.mouse_queue = queue.Queue()
        self.all_events = []  # Combined event list
        self.stop_event = threading.Event()
        self.keyboard_listener = None
        self.mouse_listener = None

    def on_key_press(self, key):
        self.keyboard_queue.put({'type': 'key_press', 'key': str(key), 'timestamp': time.time()})

    def on_key_release(self, key):
        self.keyboard_queue.put({'type': 'key_release', 'key': str(key), 'timestamp': time.time()})

    def on_mouse_click(self, x, y, button, pressed):
        self.mouse_queue.put({'type': 'mouse_click', 'x': x, 'y': y, 'button': str(button), 'pressed': pressed, 'timestamp': time.time()})

    def on_mouse_move(self, x, y):
        self.mouse_queue.put({'type': 'mouse_move', 'x': x, 'y': y, 'timestamp': time.time()})

    def event_processor(self):
        while not self.stop_event.is_set():
            while not self.keyboard_queue.empty():
                event = self.keyboard_queue.get()
                self.all_events.append(event)
            while not self.mouse_queue.empty():
                event = self.mouse_queue.get()
                self.all_events.append(event)
            time.sleep(0.01)

    def collect_data(self, duration):
        self.all_events = []  # Clear events for new collection
        self.keyboard_listener = KeyboardListener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.mouse_listener = MouseListener(on_click=self.on_mouse_click, on_move=self.on_mouse_move)
        self.keyboard_listener.start()
        self.mouse_listener.start()
        processor_thread = threading.Thread(target=self.event_processor)
        processor_thread.start()

        time.sleep(duration)
        self.stop_event.set()
        self.keyboard_listener.stop()
        self.mouse_listener.stop()
        processor_thread.join()
        return self.all_events

    def extract_features(self, events, window_size=10):
            if not events:
                return pd.DataFrame()

            df = pd.DataFrame(events)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')

            features = []
            print("--- Raw Data (Windowed) ---")  # Print raw data for each window
            for i, window in df.resample(f'{window_size}S'):
                if window.empty:
                    continue
                print(f"Window {i}:")
                print(window)  # Print the raw events in the window
                window_features = {}

                # ... (Existing feature calculation code) ...
                features.append(window_features)

            feature_df = pd.DataFrame(features)
            feature_df = feature_df.fillna(0)
            print("\n--- Calculated Features (Windowed) ---")  # Print calculated features
            print(feature_df)
            return feature_df

    def run_intrusion_detection(self, training_duration=120, inference_duration=10):
        training_events = self.collect_data(training_duration)

        print("\n--- Raw Training Data ---")  # Print all raw training data
        print(pd.DataFrame(training_events))

        training_features = self.extract_features(training_events)

        if training_features.empty:
            print("Not enough training data collected.")
            return

        scaler = StandardScaler()
        training_features_scaled = scaler.fit_transform(training_features)

        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(training_features_scaled)

        print("\n--- Scaled Training Features ---")  # Print scaled training features
        print(pd.DataFrame(training_features_scaled))

        print("\n--- Inference Results ---")
        for i in range(5):
            inference_events = self.collect_data(inference_duration)

            print("\n--- Raw Inference Data ---")  # Print raw inference data
            print(pd.DataFrame(inference_events))

            inference_features = self.extract_features(inference_events)

            if inference_features.empty:
                print("Not enough inference data collected in this window.")
                prediction = -1
            else:
                inference_features_scaled = scaler.transform(inference_features)

                print("\n--- Scaled Inference Features ---")  # Print scaled inference features
                print(pd.DataFrame(inference_features_scaled))

                prediction = clf.predict(inference_features_scaled)

            print(f"\nInference {i+1}: {'Normal' if all(p == 1 for p in prediction) else 'Anomaly Detected'}")  # Print inference output
            print("---")

def main():
    analyzer = UserBehaviorAnalyzer()
    analyzer.run_intrusion_detection()

if __name__ == "__main__":
    main()