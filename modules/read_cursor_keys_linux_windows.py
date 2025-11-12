# key_reader.py
import platform
import threading
import time

class KeyReader:
    def __init__(self):
        self._pressed_keys = []
        self._listener_started = False
        self._lock = threading.Lock()
        self._term_listener = None
        self._arrow_queue = None
        self._esc_window = 0.5
        self._esc_time = 0

        self._system = platform.system()
        if self._system == "Windows":
            try:
                import keyboard
                self._keyboard = keyboard
            except ImportError:
                self._keyboard = None
        else:
            import sys, os, termios, tty, select
            from queue import Queue, Empty
            self._sys = sys
            self._os = os
            self._termios = termios
            self._tty = tty
            self._select = select
            self._Queue = Queue
            self._Empty = Empty
            self._arrow_queue = Queue()

    # ---------------- Windows implementation
    def _on_press_windows(self, ev):
        try:
            name = ev.name
        except:
            return
        if name in ("up","down","left","right") and self._keyboard.is_pressed("esc"):
            with self._lock:
                self._pressed_keys.append(name)

    # ---------------- Linux implementation
    class _TermListener(threading.Thread):
        def __init__(self, key_reader):
            super().__init__(daemon=True)
            self._kr = key_reader
            self.running = True
            self.buf = b""

        def stop(self):
            self.running = False

        def run(self):
            fd = self._kr._sys.stdin.fileno()
            old = self._kr._termios.tcgetattr(fd)
            self._kr._tty.setcbreak(fd)
            try:
                while self.running:
                    rlist, _, _ = self._kr._select.select([self._kr._sys.stdin], [], [], 0.05)
                    if rlist:
                        data = self._kr._os.read(fd, 32)
                        if not data:
                            continue
                        self.buf += data

                    # process ESC sequences
                    i = 0
                    while i < len(self.buf):
                        b = self.buf[i]
                        if b == 0x1b:  # ESC
                            self._kr._esc_time = time.time()
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
                                    if time.time() - self._kr._esc_time <= self._kr._esc_window:
                                        self._kr._arrow_queue.put(direction)
                                i += 3
                                continue
                        i += 1
                    self.buf = self.buf[i:]
                    time.sleep(0.001)
            finally:
                self._kr._termios.tcsetattr(fd, self._kr._termios.TCSADRAIN, old)

    # ---------------- Public API
    def init_key_listener(self):
        if self._listener_started:
            return
        if self._system == "Windows":
            if self._keyboard is None:
                raise RuntimeError("Windows: `keyboard` module not available")
            self._keyboard.on_press(self._on_press_windows)
        else:
            self._term_listener = self._TermListener(self)
            self._term_listener.start()
        self._listener_started = True

    def stop_key_listener(self):
        if not self._listener_started:
            return
        if self._system == "Windows":
            if self._keyboard:
                self._keyboard.unhook_all()
        else:
            if self._term_listener:
                self._term_listener.stop()
                self._term_listener = None
        self._listener_started = False

    def read_key(self):
        if self._system == "Windows":
            if not self._listener_started:
                self.init_key_listener()
            with self._lock:
                if not self._pressed_keys:
                    return None, 0
                nUp = self._pressed_keys.count('up')
                nDown = self._pressed_keys.count('down')
                nRight = self._pressed_keys.count("right")
                nLeft = self._pressed_keys.count("left")
                n0 = self._pressed_keys.count("0")
                n1 = self._pressed_keys.count("l")
                n2 = self._pressed_keys.count("2")
                n3 = self._pressed_keys.count("3")
                n4 = self._pressed_keys.count("4")
                self._pressed_keys.clear()
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
            elif n0 > 0:
                return '0', n0
            elif n1 > 0:
                return '1', n1
            elif n2 > 0:
                return '2', n2
            elif n3 > 0:
                return '3', n3
            elif n4 > 0:
                return '4', n4            
            else:
                return None, 0
        else:
            if not self._listener_started:
                self.init_key_listener()
            keys = []
            while True:
                try:
                    keys.append(self._arrow_queue.get_nowait())
                except self._Empty:
                    break
            if not keys:
                return None, 0
            counts = {k: keys.count(k) for k in ('up','down','left','right','0','1','2','3','4')}
            direction, count = max(counts.items(), key=lambda x: x[1])
            return direction, count

# ---------------- Demo
if __name__ == "__main__":
    kr = KeyReader()
    kr.init_key_listener()
    print("Press ESC + arrow keys. Ctrl-C to exit.")
    try:
        while True:
            d, c = kr.read_key()
            if d:
                print(f"{d} pressed {c} times")
            time.sleep(2.5)
    except KeyboardInterrupt:
        kr.stop_key_listener()
        print("\nStopped.")
