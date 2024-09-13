import torch
from torchvision import transforms
import numpy as np
from src.Deepixbis_model import DeePixBiS

model = DeePixBiS()
model.load_state_dict(torch.load('models/faceantispoof/DeePixBiS.pth'))
# ckpt = torch.load('Pretrained_models/RM_grandtest_model_0_0.pth')
# model.load_state_dict(ckpt['state_dict'])
model.eval()

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_facespoof(frame, bbox):
    # faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
    # cv.imshow('Test', faceRegion)
    x, y, w, h = bbox
    faceRegion = frame[y:y + h, x:x + w]
    faceRegion = tfms(faceRegion)
    try:
        faceRegion = faceRegion.unsqueeze(0)
    except:
        faceRegion = np.array(faceRegion)
        faceRegion = torch.tensor(faceRegion)
        faceRegion = faceRegion.float()
        faceRegion = faceRegion.unsqueeze(0)

    mask, binary = model.forward(faceRegion)
    res = torch.mean(mask).item()

    return res

if __name__ == "__main__":
    
    import cv2
    from face_detection import detect_face

    cap = cv2.VideoCapture(0)
    i=0
    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_box = detect_face(rgb_frame)
        res = predict_facespoof(rgb_frame, face_box[0])
        
        cv2.putText(frame, "Face Spoof: "+str(round(res,3)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Keypoints', frame)
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
