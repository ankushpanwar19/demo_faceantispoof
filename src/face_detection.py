
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


def detect_face(rgb_frame):
    image_height, image_width, _ = rgb_frame.shape
    detection_results = face_detection.process(rgb_frame)

    face_box = []
    no_of_faces = ''
    if detection_results.detections is None:
        no_of_faces = 'no_face'
    elif detection_results.detections is not None and len(detection_results.detections)==1:
        no_of_faces = 'single_face'
        for detection in detection_results.detections:
            # Retrieve bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            bbox_x = int(bbox.xmin * image_width)
            bbox_y = int(bbox.ymin * image_height)
            bbox_width = int(bbox.width * image_width)
            bbox_height = int(bbox.height * image_height)
    
            face_box = [bbox_x,bbox_y,bbox_width,bbox_height]
    elif len(detection_results.detections)>1:
        no_of_faces = 'mulitple_face'

    return face_box, no_of_faces

def detect_landmarks(rgb_frame):

    landmarks = []
    image_height, image_width, _ = rgb_frame.shape
    mesh_results = face_mesh.process(rgb_frame)
    face_landmark = mesh_results.multi_face_landmarks[0]
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            landmarks = [(int(pt.x * image_width), int(pt.y * image_height)) for pt in face_landmarks.landmark]

    return landmarks, face_landmark

if __name__ == "__main__":
    import cv2
    im_path = "data/images/phone.png"
    img = cv2.imread(im_path)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_box, no_of_faces = detect_face(rgb_frame)
    landmarks =  detect_landmarks(rgb_frame)
    print(face_box)
    print(len(landmarks))