import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eye landmarks (MediaPipe indices)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Dynamic calibration variables
calibration_frames = 30  # Number of frames for calibration
left_ear_history = []
right_ear_history = []
calibrated = False
left_thresholds = {'closed': 0.15, 'partial': 0.25}
right_thresholds = {'closed': 0.15, 'partial': 0.25}

def calculate_ear(eye_landmarks):
    # Vertical distances
    d1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                         np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    d2 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - 
                         np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    # Horizontal distance
    d3 = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                         np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    ear = (d1 + d2) / (2.0 * d3)
    return ear

def calibrate_thresholds(left_ear_values, right_ear_values):
    """Calculate adaptive thresholds based on collected EAR values"""
    left_avg = np.mean(left_ear_values)
    right_avg = np.mean(right_ear_values)
    
    # Set thresholds as percentages of average open eye EAR
    # Lower "closed" threshold so partial states are detected better
    left_thresholds = {
        'closed': left_avg * 0.45,    # 35% of average for closed (was 40%)
        'partial': left_avg * 0.65    # 65% of average for partial
    }
    
    right_thresholds = {
        'closed': right_avg * 0.20,   # 20% of average for closed (was 40%)
        'partial': right_avg * 0.65   # 65% of average for partial
    }
    
    return left_thresholds, right_thresholds

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    # Process image with MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left eye landmarks
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
            left_ear = calculate_ear(left_eye)
            
            # Get right eye landmarks
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]
            right_ear = calculate_ear(right_eye)
            
            # Calibration phase
            if not calibrated:
                left_ear_history.append(left_ear)
                right_ear_history.append(right_ear)
                
                # Display calibration progress
                cv2.putText(image, f"CALIBRATING... Keep eyes OPEN", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Progress: {len(left_ear_history)}/{calibration_frames}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if len(left_ear_history) >= calibration_frames:
                    left_thresholds, right_thresholds = calibrate_thresholds(left_ear_history, right_ear_history)
                    calibrated = True
                    print(f"Calibration complete!")
                    print(f"Left thresholds: {left_thresholds}")
                    print(f"Right thresholds: {right_thresholds}")
            
            else:
                # Normal detection phase
                # Classify left eye state
                if left_ear < left_thresholds['closed']:
                    left_eye_state = "CLOSED"
                elif left_ear < left_thresholds['partial']:
                    left_eye_state = "PARTIALLY OPEN"
                else:
                    left_eye_state = "OPEN"
                
                # Classify right eye state
                if right_ear < right_thresholds['closed']:
                    right_eye_state = "CLOSED"
                elif right_ear < right_thresholds['partial']:
                    right_eye_state = "PARTIALLY OPEN"
                else:
                    right_eye_state = "OPEN"
                
                # Display both eye states
                cv2.putText(image, f"Left Eye: {left_eye_state}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Right Eye: {right_eye_state}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display EAR values and thresholds
                cv2.putText(image, f"Left EAR: {left_ear:.2f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(image, f"Right EAR: {right_ear:.2f}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw eye landmarks
            for eye_indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
                for index in eye_indices:
                    landmark = face_landmarks.landmark[index]
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow('Eye Detection', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
