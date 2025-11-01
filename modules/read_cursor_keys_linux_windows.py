# key_reader.py
import platform
import threading
import time

pressed_keys = []
_listener_started = False
_lock = threading.Lock()

# ------------------ Windows
if platform.system() == "Windows":
    try:
        import keyboard
    except ImportError:
        keyboard = None

    def _on_press_windows(ev):
        try:
            name = ev.name
        except:
            return
        if name in ("up","down","left","right") and keyboard.is_pressed("esc"):
            with _lock:
                pressed_keys.append(name)

    def init_key_listener():
        global _listener_started
        if _listener_started:
            return
        if keyboard is None:
            raise RuntimeError("Windows: `keyboard` module not available")
        keyboard.on_press(_on_press_windows)
        _listener_started = True

    def stop_key_listener():
        global _listener_started
        if keyboard:
            keyboard.unhook_all()
        _listener_started = False

# ------------------ Linux
else:
    import sys, os, termios, tty, select
    from queue import Queue, Empty

    _ESC_WINDOW = 0.5  # ignore ESC sequences older than this (for ESC+arrow)
    _esc_time = 0
    _arrow_queue = Queue()

    class _TermListener(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.running = True
            self.buf = b""

        def stop(self):
            self.running = False

        def run(self):
            global _esc_time
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while self.running:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if rlist:
                        data = os.read(fd, 32)
                        if not data:
                            continue
                        self.buf += data

                    # process ESC sequences
                    i = 0
                    while i < len(self.buf):
                        b = self.buf[i]
                        if b == 0x1b:  # ESC
                            _esc_time = time.time()
                            # check for arrow sequence
                            if i+2 < len(self.buf) and self.buf[i+1] == 0x5b:
                                seq = self.buf[i+2]
                                direction = None
                                if seq == 0x41:
                                    direction = 'up'
                                elif seq == 0x42:
                                    direction = 'down'
                                elif seq == 0x43:
                                    direction = 'right'
                                elif seq == 0x44:
                                    direction = 'left'
                                if direction:
                                    # only count if within ESC_WINDOW
                                    if time.time() - _esc_time <= _ESC_WINDOW:
                                        _arrow_queue.put(direction)
                                i += 3
                                continue
                        i += 1
                    self.buf = self.buf[i:]
                    time.sleep(0.001)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    _term_listener = None

    def init_key_listener():
        global _listener_started, _term_listener
        if _listener_started:
            return
        _term_listener = _TermListener()
        _term_listener.start()
        _listener_started = True

    def stop_key_listener():
        global _listener_started, _term_listener
        if _term_listener:
            _term_listener.stop()
            _term_listener = None
        _listener_started = False

# ------------------ common read_key
def read_key():
    global pressed_keys
    if platform.system() == "Windows":
        if not _listener_started:
            init_key_listener()
        with _lock:
            if not pressed_keys:
                return None, 0
            nUp = pressed_keys.count('up')
            nDown = pressed_keys.count('down')
            nRight = pressed_keys.count("right")
            nLeft = pressed_keys.count("left")
            pressed_keys.clear()
        deltaUpDown = nUp - nDown
        deltaRightLeft = nRight - nLeft
        if deltaUpDown > 0:
            return 'up', deltaUpDown
        elif deltaUpDown < 0:
            return 'down', -deltaUpDown
        elif deltaRightLeft > 0:
            return 'right', deltaRightLeft
        elif deltaRightLeft < 0:
            return 'left', -deltaRightLeft
        else:
            return None, 0
    else:
        # Linux
        if not _listener_started:
            init_key_listener()
        keys = []
        while True:
            try:
                keys.append(_arrow_queue.get_nowait())
            except Empty:
                break
        if not keys:
            return None, 0
        counts = {k: keys.count(k) for k in ('up','down','left','right')}
        # return the most pressed direction
        direction, count = max(counts.items(), key=lambda x:x[1])
        return direction, count

# ------------------ demo
if __name__ == "__main__":
    print("Press ESC + arrow keys...")
    init_key_listener()
    try:
        while True:
            d, c = read_key()
            if d:
                print(f"{d} pressed {c} times")
            time.sleep(2.5)
    except KeyboardInterrupt:
        stop_key_listener()
        print("\nStopped.")
