
import inspect
import os.path as osp
import sys

import cv2 as cv
import numpy as np

current_dir = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = osp.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.lightantispoof.model import TorchCNN
import src.lightantispoof.utils as utils

config = "models/lightantispoof/config.py"
device = "cpu"
spoof_model_path = "models/lightantispoof/MN3_antispoof.pth.tar"
if not config:
    raise ValueError('You should pass config file to work with a Pytorch model')
config = utils.read_py_config(config)
spoof_model = utils.build_model(config, device, strict=True, mode='eval')
spoof_model = TorchCNN(spoof_model, spoof_model_path, config, device=device)


def light_predict_facespoof(rgb_frame, face_bbox):
    """Get prediction for all detected faces on the frame"""
    # faces = []
    # print("Facebox",face_bbox)
    x, y, w, h = face_bbox
    face = rgb_frame[y:y + h, x:x + w]
    faces = np.array([face])
    # print("Shape:",faces.shape)
    output = None, None
    output = spoof_model.forward(faces)
    output = list(map(lambda x: x.reshape(-1), output))
    return output[0][1]

if __name__ == '__main__':
    
    import cv2
    from face_detection import detect_face

    cap = cv2.VideoCapture(0)
    i=0
    res = 0
    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_box, no_of_faces = detect_face(rgb_frame)
        
        if no_of_faces == "single_face":
            # print("Shape:",faces.shape)
            res = light_predict_facespoof(rgb_frame, face_box)
            # print(res)
        
        cv2.putText(frame, "Face Spoof: "+str(round(res[0][0],3)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Keypoints', frame)
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
