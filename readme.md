
# Sentinel: User Behaviour Analysis for Real Time Intrusion Detection 🔐


## 📝 Overview
Our software solution revolutionizes authentication by continuously analyzing user behavior across the system to ensure genuine access. It monitors activities and flags any deviation from recognized patterns, effectively detecting intruders—even if they bypass passwords. Once flagged, system-wide activities are blocked until the user successfully re-authenticates via Google Authenticator, guaranteeing security through behavioral analysis.


## ✨ Features
- **Continuous Monitoring:** Constantly tracks cursor behavior, keypress dynamics, active windows, focus sessions, and scroll behavior.
- **Browser Activity Tracking:** Monitors tab switching and interactions via a dedicated web extension.
- **Intruder Detection:** Uses an Isolation Forest model to flag anomalous user behavior.
- **Adaptive Learning:** Incorporates incremental learning with a sliding window technique to evolve with user behavior.
- **User-specific Security:** Supports multiple users and a customizable safe list for file sharing.
- **Local Processing:** All operations occur locally, ensuring enhanced security.
- **Real-time Reporting:** Generates dynamic reports using a local LLM (Llama via Ollama).

## ⚙️ Components and Working
- **System Information Fetcher:** Collects low-level metrics like CPU usage, memory stats, and system-level activity (cursor position, keypress duration, timestamps).
- **Browser Information Fetcher:** Captures browser-specific activities such as tab switching, interaction with interactive elements, and browser metadata.
- **ML Pipeline:** Transforms low-level features into high-level features (e.g., cursor speed standard deviation, character rate, dwell time, flight time) and runs the Isolation Forest model for anomaly detection.
- **System Blocker and Authenticator:** Blocks system access upon intruder detection and initiates a re-authentication process through the integrated Google Authenticator API.
- **LLM Summarizer:** Utilizes Llama running locally with Ollama to generate real-time reports summarizing user activities and security alerts.

## 📊 Results
The system successfully predicts intruders with approximately 80% accuracy, ensuring robust protection even in environments with multiple users and evolving user behaviors.

## 🛠️ Installation and usage
*Details coming soon...*

## Technical Details
*Details coming soon...*

## 👨‍💻 Author - Achintya Varshneya
I'm fluent in Python, TensorFlow, and the ancient art of hyperparameter tuning. If you hear strange chanting coming from my office, it's probably just me trying to summon a better learning rate... or maybe just trying to convince my GPU to stop overheating. Either way, please bring coffee. 
