import cv2
import mediapipe as mp
import pyautogui

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

# Track key states
key_state = {"left": False, "right": False, "up": False, "down": False}

def hold_key(key, state):
    """Hold or release keys smoothly"""
    if state and not key_state[key]:
        pyautogui.keyDown(key)
        key_state[key] = True
    elif not state and key_state[key]:
        pyautogui.keyUp(key)
        key_state[key] = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip for mirror effect
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Process frame
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Default → release keys
    move_left = move_right = move_up = move_down = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Index fingertip
        x, y = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
        cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

        # Define movement zones
        left_zone = w // 3
        right_zone = 2 * w // 3
        top_zone = h // 3
        bottom_zone = 2 * h // 3

        # Left/Right detection
        if x < left_zone:
            move_left = True
            cv2.putText(image, "LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        elif x > right_zone:
            move_right = True
            cv2.putText(image, "RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Up/Down detection
        if y < top_zone:
            move_up = True
            cv2.putText(image, "UP", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        elif y > bottom_zone:
            move_down = True
            cv2.putText(image, "DOWN", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    # Smooth key control
    hold_key("left", move_left)
    hold_key("right", move_right)
    hold_key("up", move_up)
    hold_key("down", move_down)

    cv2.imshow("Hand Game Control", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all keys
for k in key_state:
    if key_state[k]:
        pyautogui.keyUp(k)

cap.release()
cv2.destroyAllWindows()
