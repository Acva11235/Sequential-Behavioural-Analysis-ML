 def get_active_window_info():
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            exe = psutil.Process(pid).exe()
        except:
            exe = "unknown"
        return title, exe, hwnd

    def get_all_open_windows():
        windows = []
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                windows.append((hwnd, title))
        win32gui.EnumWindows(callback, None)
        return windows

    def update(self):
        # Track focused app
        title, exe, hwnd = self.get_active_window_info()
        if hwnd != self.current_focus:
            if self.current_focus is not None:
                self.focus_history[-1]["end"] = time.time()
            self.focus_history.append({
                "title": title,
                "exe": exe,
                "start": time.time(),
                "end": None
            })
            self.current_focus = hwnd
        
        Track all open apps every 10s
        if time.time() - self.last_scan > 10:
            self._track_open_apps()
            self.last_scan = time.time()
            
    def _track_open_apps(self):
        current_windows = self.get_all_open_windows()
        current_hwnds = {hwnd for hwnd, _ in current_windows}
        
        # Detect closed apps
        for hwnd in list(self.open_apps.keys()):
            if hwnd not in current_hwnds:
                self._record_app_close(hwnd)
        
        # Detect new apps
        for hwnd, title in current_windows:
            if hwnd not in self.open_apps:
                self._record_app_open(hwnd, title)
                
    def _record_app_open(self, hwnd, title):
        self.open_apps[hwnd] = {
            "title": title,
            "exe": self._get_exe_path(hwnd),
            "start": time.time(),
            "end": None
        }
    
    def _record_app_close(self, hwnd):
        if hwnd in self.open_apps:
            self.open_apps[hwnd]["end"] = time.time()