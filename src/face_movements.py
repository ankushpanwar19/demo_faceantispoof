import numpy as np

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 191, 80, 81, 13, 311, 308, 324, 87, 178,14]
left_eye_lid = [159, 145] #[upperlid,lowerlid]
right_eye_lid = [386, 374] #[upperlid,lowerlid]
left_eye_corners = [33,133]
right_eye_corners = [362,263]

def calculate_ear(landmarks, eye_indices):
     # Convert landmarks to NumPy arrays (x, y) for the eye
    p1 = np.array(landmarks[eye_indices[0]])  # horizontal start point
    p2 = np.array(landmarks[eye_indices[1]])  # upper vertical
    p3 = np.array(landmarks[eye_indices[2]])  # upper vertical
    p4 = np.array(landmarks[eye_indices[3]])  # horizontal end point
    p5 = np.array(landmarks[eye_indices[4]])  # lower vertical
    p6 = np.array(landmarks[eye_indices[5]])  # lower vertical

    # Compute vertical distances
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    # Compute horizontal distance
    horizontal = np.linalg.norm(p1 - p4)

    # Compute the EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def eye_blink_ear(landmarks, blink_count=0, prev_eyes=''):
    EAR_THRESHOLD = 0.22
    left_eye_indices = [133, 158, 159, 33, 145, 153]
    right_eye_indices = [362, 385, 386, 263, 380, 374]
    left_eye_ear = calculate_ear(landmarks,left_eye_indices)
    right_eye_ear = calculate_ear(landmarks,right_eye_indices)
    
    # print("Eye:",left_eye_ear,left_eye_ear)
    if (left_eye_ear<EAR_THRESHOLD ) or (right_eye_ear<EAR_THRESHOLD):
        if prev_eyes == '':
            prev_eyes = 'closed'
            blink_count +=1
        elif prev_eyes == 'opened':
            blink_count +=1
            prev_eyes = 'closed'

    else:
        if prev_eyes == '':
            prev_eyes = 'opened'
        elif prev_eyes == 'closed':
            blink_count +=1
            prev_eyes = 'opened'

    return blink_count, prev_eyes

def eye_blink(landmarks, blink_count=0, prev_eyes=''):
    
    left_x_distance =np.linalg.norm(np.array(landmarks[left_eye_corners[0]]) - np.array(landmarks[left_eye_corners[1]]))
    right_x_distance =np.linalg.norm(np.array(landmarks[right_eye_corners[0]]) - np.array(landmarks[right_eye_corners[1]]))
    left_y_distance =np.linalg.norm(np.array(landmarks[left_eye_lid[0]]) - np.array(landmarks[left_eye_lid[1]]))
    right_y_distance =np.linalg.norm(np.array(landmarks[right_eye_lid[0]]) - np.array(landmarks[right_eye_lid[1]]))

    left_ratio = left_x_distance/left_y_distance
    right_ratio = right_x_distance/right_y_distance
    # print(f"Left: {left_ratio}, Right :{right_ratio}")
    if (left_ratio>4 )or (right_ratio>4):
        if prev_eyes == '':
            prev_eyes = 'closed'
            blink_count +=1
        elif prev_eyes == 'opened':
            blink_count +=1
            prev_eyes = 'closed'

    else:
        if prev_eyes == '':
            prev_eyes = 'opened'
        elif prev_eyes == 'closed':
            blink_count +=1
            prev_eyes = 'opened'
    # print(blink_count)                        
    return blink_count, prev_eyes

def mouth_movement(landmarks, mouth_count=0, prev_mouth=''):

    lip_y_distance = np.linalg.norm(np.array(landmarks[MOUTH[4]]) - np.array(landmarks[MOUTH[10]]))
    lip_x_distance = np.linalg.norm(np.array(landmarks[62]) - np.array(landmarks[308]))

    lip_ratio = lip_x_distance/lip_y_distance
    # if mar > MAR_THRESHOLD and lip_distance>5:
    # print(lip_ratio)
    if lip_ratio>20:
        if prev_mouth == '':
            prev_mouth = 'closed'
        elif prev_mouth == 'opened':
            mouth_count +=1
            prev_mouth = 'closed'

    else:
        if prev_mouth == '':
            prev_mouth = 'opened'
        elif prev_mouth == 'closed':
            mouth_count +=1
            prev_mouth = 'opened'

    return mouth_count, prev_mouth

if __name__ == "__main__":
    
    import cv2
    from face_detection import detect_landmarks

    cap = cv2.VideoCapture(0)
    i=0
    blink_count = 0
    prev_eyes = ''
    mouth_count = 0
    prev_mouth = ''
    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detect_landmarks(rgb_frame)

        blink_count, prev_eyes = eye_blink(landmarks,blink_count,prev_eyes)
        mouth_count, prev_mouth = mouth_movement(landmarks,mouth_count,prev_mouth)

        cv2.putText(frame, "Blink Count: "+str(blink_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Mouth Count: "+str(mouth_count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Keypoints', frame)
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
