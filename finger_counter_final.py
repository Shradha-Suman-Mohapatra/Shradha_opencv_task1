import cv2 
import mediapipe as mp

# Initialize mediapipe hands and drawing
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Define fingertip landmark IDs (Thumb, Index, Middle, Ring, Pinky)
finger_tips = [4, 8, 12, 16, 20]

while cap.isOpened(): 
    success, image = cap.read() 
    if not success: 
        break

    # Flip the image for mirror view and convert BGR to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(rgb_image)

    finger_count = 0  # Default count

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            landmarks = hand_landmarks.landmark
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'

            # Count raised fingers (excluding thumb)
            for tip_id in finger_tips[1:]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    finger_count += 1

            # Thumb logic (based on hand label)
            if hand_label == "Right":
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
                    finger_count += 1
            else:  # Left hand
                if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                    finger_count += 1

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show finger count on screen
    cv2.putText(image, f'Fingers: {finger_count}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the output
    cv2.imshow("Finger Counter", image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release() 
cv2.destroyAllWindows()