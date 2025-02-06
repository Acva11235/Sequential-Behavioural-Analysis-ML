import os
import sqlite3
from datetime import timedelta
import datetime

def get_time_window():
    """
    Returns the current time and the time 30 seconds ago in the correct string format
    """
    current_time = datetime.now()
    time_window = current_time - timedelta(seconds=30)
    return time_window.strftime('%Y-%m-%d %H:%M:%S'), current_time.strftime('%Y-%m-%d %H:%M:%S')

def extract_key_inference():
    DB_PATH = os.path.join(os.path.expanduser("~"), "Documents", "soft_activity.sqlite")
    start_time, end_time = get_time_window()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT key, key_interval, timestamp 
        FROM SOFTWARE 
        WHERE TYPE = "Keyboard"
        AND timestamp BETWEEN ? AND ?;
    """, (start_time, end_time))
    
    result = cursor.fetchall()
    conn.close()
    
    return result

def extract_mouse_inference():
    DB_PATH = os.path.join(os.path.expanduser("~"), "Documents", "soft_activity.sqlite")
    start_time, end_time = get_time_window()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT click_type, click_interval, position, timestamp 
        FROM SOFTWARE 
        WHERE TYPE = "Click"
        AND timestamp BETWEEN ? AND ?;
    """, (start_time, end_time))
    
    result = cursor.fetchall()
    conn.close()
    
    return result

def extract_focus_inference():
    DB_PATH = os.path.join(os.path.expanduser("~"), "Documents", "soft_activity.sqlite")
    start_time, end_time = get_time_window()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT title, duration, timestamp 
        FROM SOFTWARE 
        WHERE TYPE = "App in Focus"
        AND timestamp BETWEEN ? AND ?;
    """, (start_time, end_time))
    
    result = cursor.fetchall()
    conn.close()
    
    return result