import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=r'C:\Codes\ESP32-HomeAutomation\AI-CRAFT-Codebase\gesture-control\hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Gesture tracking variables
cooldown_time = 1.0  # Seconds between gestures
last_gesture_time = 0
last_gesture_type = None


def lights_on():
    """Function called when showing OPEN HAND - Turn all lights ON"""
    print("🟢 All Lights ON - Open Hand")
    # Add your custom action here


def lights_off():
    """Function called when showing CLOSED FIST - Turn all lights OFF"""
    print("🔴 All Lights OFF - Closed Fist")
    # Add your custom action here


def fan_toggle():
    """Function called when showing 1 FINGER - Toggle Fan"""
    print("☝️ Fan ON/OFF - 1 Finger")
    # Add your custom action here


def ac_toggle():
    """Function called when showing 2 FINGERS - Toggle AC"""
    print("✌️ AC ON/OFF - 2 Fingers")
    # Add your custom action here


def tv_toggle():
    """Function called when showing 3 FINGERS - Toggle TV"""
    print("🤙 TV ON/OFF - 3 Fingers")
    # Add your custom action here


def calculate_angle(point_a, point_b, point_c):
    """Calculate the angle formed by three landmarks."""
    vector_ab = np.array([point_a.x - point_b.x, point_a.y - point_b.y])
    vector_cb = np.array([point_c.x - point_b.x, point_c.y - point_b.y])

    norm_ab = np.linalg.norm(vector_ab)
    norm_cb = np.linalg.norm(vector_cb)
    if norm_ab == 0 or norm_cb == 0:
        return 0.0

    cosine_angle = np.dot(vector_ab, vector_cb) / (norm_ab * norm_cb)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def get_finger_states(hand_landmarks, handedness_label=None):
    """Return which fingers are extended."""
    thumb_angle = calculate_angle(hand_landmarks[4], hand_landmarks[3], hand_landmarks[2])
    thumb_tip_to_index_base = np.hypot(
        hand_landmarks[4].x - hand_landmarks[5].x,
        hand_landmarks[4].y - hand_landmarks[5].y,
    )
    thumb_joint_to_index_base = np.hypot(
        hand_landmarks[3].x - hand_landmarks[5].x,
        hand_landmarks[3].y - hand_landmarks[5].y,
    )

    if handedness_label == "Right":
        thumb_direction_ok = hand_landmarks[4].x < hand_landmarks[3].x
    elif handedness_label == "Left":
        thumb_direction_ok = hand_landmarks[4].x > hand_landmarks[3].x
    else:
        thumb_direction_ok = abs(hand_landmarks[4].x - hand_landmarks[3].x) > 0.02

    states = {
        "thumb": (
            thumb_angle > 150
            and thumb_tip_to_index_base > thumb_joint_to_index_base * 1.15
            and thumb_direction_ok
        ),
        "index": hand_landmarks[8].y < hand_landmarks[6].y < hand_landmarks[5].y,
        "middle": hand_landmarks[12].y < hand_landmarks[10].y < hand_landmarks[9].y,
        "ring": hand_landmarks[16].y < hand_landmarks[14].y < hand_landmarks[13].y,
        "pinky": hand_landmarks[20].y < hand_landmarks[18].y < hand_landmarks[17].y,
    }
    return states


def count_fingers(hand_landmarks, handedness_label=None):
    """
    Count extended fingers
    Returns: number of extended fingers (0-5)
    """
    return sum(get_finger_states(hand_landmarks, handedness_label).values())


def detect_gesture(hand_landmarks, current_time, handedness_label=None):
    """
    Detect hand open, closed, or exact finger-count gestures
    Returns: 'open', 'closed', or None
    """
    global last_gesture_time, last_gesture_type
    
    if hand_landmarks is None:
        return None
    
    # Check cooldown
    if current_time - last_gesture_time < cooldown_time:
        return None

    finger_states = get_finger_states(hand_landmarks, handedness_label)
    fingers = sum(finger_states.values())

    if fingers == 5 and last_gesture_type != 'open':
        last_gesture_time = current_time
        last_gesture_type = 'open'
        return 'open'

    elif fingers == 0 and last_gesture_type != 'closed':
        last_gesture_time = current_time
        last_gesture_type = 'closed'
        return 'closed'

    elif fingers == 1 and (finger_states['thumb'] or finger_states['index']) and last_gesture_type != 'one':
        last_gesture_time = current_time
        last_gesture_type = 'one'
        return 'one'
        
    elif fingers == 2 and last_gesture_type != 'two':
        last_gesture_time = current_time
        last_gesture_type = 'two'
        return 'two'
    
    elif fingers == 3 and last_gesture_type != 'three':
        last_gesture_time = current_time
        last_gesture_type = 'three'
        return 'three'
                
    return None


def draw_landmarks_on_image(frame, detection_result):
    """Draw hand landmarks on the image"""
    if not detection_result.hand_landmarks:
        return frame
    
    # Hand landmark connections (simplified)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 17), (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    ]
    
    for hand_landmarks in detection_result.hand_landmarks:
        h, w, _ = frame.shape
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
    
    return frame


def main():
    """Main function to run gesture control"""
    global last_gesture_time
    cap = cv2.VideoCapture(0)
    
    print("Hand Gesture Control Started!")
    print("✋ OPEN HAND (5 fingers) - All Lights ON")
    print("✊ CLOSED FIST (0 fingers) - All Lights OFF")
    print("☝️ 1 FINGER - Fan ON/OFF")
    print("✌️ 2 FINGERS - AC ON/OFF")
    print("🤙 3 FINGERS - TV ON/OFF")
    print("Press 'q' to quit\n")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        current_time = time.time()
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = landmarker.detect(mp_image)
        
        hand_landmarks = None
        handedness_label = None
        fingers_count = 0
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            if detection_result.handedness and detection_result.handedness[0]:
                handedness_label = detection_result.handedness[0][0].category_name
            fingers_count = count_fingers(hand_landmarks, handedness_label)
            
            # Draw landmarks on frame
            frame = draw_landmarks_on_image(frame, detection_result)
            
            # Draw finger count
            cv2.putText(frame, f"Fingers: {fingers_count}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        # Draw gesture instructions with icons
        # Open hand icon area
        cv2.rectangle(frame, (10, 130), (200, 230), (0, 255, 0), 2)
        cv2.putText(frame, "OPEN HAND", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "= ON", (60, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Closed fist icon area
        cv2.rectangle(frame, (frame_width - 200, 130), (frame_width - 10, 230), (0, 0, 255), 2)
        cv2.putText(frame, "CLOSED FIST", (frame_width - 190, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "= OFF", (frame_width - 130, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Detect gesture
        gesture = detect_gesture(hand_landmarks, current_time, handedness_label)
        
        # Show cooldown timer
        time_since_last = current_time - last_gesture_time
        if time_since_last < cooldown_time:
            remaining = cooldown_time - time_since_last
            cv2.putText(frame, f"Wait: {remaining:.1f}s", (frame_width - 180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if gesture == 'open':
            lights_on()
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 10)
            cv2.putText(frame, "ON - OPEN HAND", (frame_width//2 - 200, frame_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            
        elif gesture == 'closed':
            lights_off()
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 255), 10)
            cv2.putText(frame, "OFF - CLOSED FIST", (frame_width//2 - 220, frame_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
        elif gesture == 'one':
            fan_toggle()
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (255, 165, 0), 10)
            cv2.putText(frame, "FAN ON/OFF - 1 FINGER", (frame_width//2 - 240, frame_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 165, 0), 4)
            
        elif gesture == 'two':
            ac_toggle()
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 165, 255), 10)
            cv2.putText(frame, "AC ON/OFF - 2 FINGERS", (frame_width//2 - 230, frame_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
            
        elif gesture == 'three':
            tv_toggle()
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (255, 0, 255), 10)
            cv2.putText(frame, "TV ON/OFF - 3 FINGERS", (frame_width//2 - 230, frame_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

        # Display instructions
        cv2.putText(frame, "Show OPEN HAND or CLOSED FIST to control", 
                   (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Hand Gesture Control', frame)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\nGesture control stopped.")


if __name__ == "__main__":
    main()
