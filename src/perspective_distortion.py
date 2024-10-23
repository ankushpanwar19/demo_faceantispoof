import numpy as np

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# def perspective_transformation(landmarks):
#     left_eye = landmarks[263]
#     right_eye = landmarks[33]
#     nose_tip = landmarks[4]
#     eye_distance = euclidean_distance(left_eye, right_eye)
#     nose_to_eye_distance = euclidean_distance(nose_tip, ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2))

#     ratio = round(eye_distance/nose_to_eye_distance,4)

#     return ratio

def perspective_transformation(landmarks):


    landmark = [(pt.x, pt.y, pt.z) for pt in landmarks.landmark]
    left_eye = landmark[263]
    right_eye = landmark[33]
    left_ear = landmark[356]
    right_ear = landmark[127]
    print("left_ear",left_ear)
    eye_distance = euclidean_distance(left_eye, right_eye)
    ear_distance = euclidean_distance(left_ear, right_ear)

    ratio = round(ear_distance/eye_distance,4)

    return ratio


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

        if len(landmarks)>0:
            ratio = perspective_transformation(landmarks)
        else:
            ratio = "No face"
        cv2.putText(frame, "Perspective eyes/nose : "+str(ratio), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Keypoints', frame)
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
