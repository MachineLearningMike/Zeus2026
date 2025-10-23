# key_reader.py
import keyboard

pressed_keys = []
_listener_started = False

def _on_press(key):
    # React only if UP/DOWN is pressed while ESC is held
    if key.name in ("up", "down", "left", "right") and keyboard.is_pressed("esc"):
        pressed_keys.append(key.name)

def init_key_listener():
    """Call this once to start the background listener"""
    global _listener_started
    if not _listener_started:
        keyboard.on_press(_on_press)
        _listener_started = True

def read_key():
    """
    Non-blocking.
    Returns a tuple (direction, delta):
      - direction: 'up' or 'down', or None if delta == 0
      - delta: nUp - nDown
    Clears the pressed keys after reading.
    """
    global pressed_keys
    if not _listener_started:
        init_key_listener()
    if not pressed_keys:
        return None, 0

    nUp = pressed_keys.count('up')
    nDown = pressed_keys.count('down')
    nRight = pressed_keys.count("right")
    nLeft = pressed_keys.count("left")
    deltaUpDown = nUp - nDown
    deltaRightLeft = nRight - nLeft

    # Clear stored keys after reading
    pressed_keys.clear()

    if deltaUpDown > 0:
        return 'up', deltaUpDown
    elif deltaUpDown < 0:
        return 'down', -deltaUpDown
    # We lose simultaneous vertical and horizontal moves.
    elif deltaRightLeft > 0:
        return 'right', deltaRightLeft
    elif deltaRightLeft < 0:
        return 'left', -deltaRightLeft
    else:
        return None, 0


if __name__ == "__main__":
    print("Press ESC + UP / ESC + DOWN...")
    while True:
        direction, count = read_key()
        if direction:
            print(f"{direction} pressed {count} times")
