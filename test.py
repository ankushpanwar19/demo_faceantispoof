from src.face_detection import detect_landmarks
from src.face_antispoof import predict_facespoof
from src.object_detection import run_detector, load_objdetection_model
from src.perspective_distortion import perspective_transformation
from src.face_movements import eye_blink,mouth_movement

OBJECT_DETECTION = True
FACE_ANTISPOOF = True
FACE_MOVEMENT = True
PERSPECTIVE_DISTORTION = True

# blink_count = 0
# prev_eyes = ''
# mouth_count = 0
# prev_mouth = ''

def multimodal_antispoof(rgb_frame,face_bbox):

    image_height, image_width, _ = rgb_frame.shape
    bbox_x,bbox_y,bbox_width,bbox_height = face_bbox
    responses = []
    responses_color = []
    color_code = {"Green":(0, 255, 0), "Red":(0,0,255)}
    if FACE_ANTISPOOF:
        res = predict_facespoof(rgb_frame,face_bbox)
        res = round(res,2)
        out = "Real" if res>0.6 else "Spoof"
        color = "Green" if out=='Real' else "Red"
        responses.append(f"Face Antispoof:{out}({res})")
        responses_color.append(color_code[color])
    
    if OBJECT_DETECTION:
        obj_spoof = "Real"
        obj_detected, obj_bbox = run_detector(rgb_frame)
        responses.append(f"Object Detected:{obj_detected}")
        responses_color.append(color_code["Green"])
        for i, obj in enumerate(obj_detected):
            if obj in ['laptop','tv','cellphone','book','remote']:
                ymin,xmin,ymax,xmax = obj_bbox[i]
                margin = 10
                a = bbox_x >= max(0,xmin-margin)
                b = bbox_y >= max(0,ymin-margin)
                c = min(image_width, bbox_x + bbox_width) <= min(image_width, xmax+margin)
                d = min(image_height, bbox_y + bbox_height) <= min(image_height, ymax+margin)
                if a and b and c and d:
                    obj_spoof = "Spoof"
                    break
                
        out = obj_spoof
        color = "Green" if out=='Real' else "Red"
        responses.append(f"Object Antispoof:{out}")
        responses_color.append(color_code[color])
    
    

    return responses, responses_color


def movement(rgb_frame,blink_count,mouth_count,prev_eyes,prev_mouth):
    responses = []
    responses_color = []
    color_code = {"Green":(0, 255, 0), "Red":(0,0,255)}
    if FACE_MOVEMENT:
        landmarks = detect_landmarks(rgb_frame)
        blink_count, prev_eyes = eye_blink(landmarks,blink_count,prev_eyes)
        mouth_count, prev_mouth = mouth_movement(landmarks,mouth_count,prev_mouth)

        # print(blink_count, prev_eyes)
        color = "Green"
        responses.append(f"Eye Blink:{blink_count}")
        responses_color.append(color_code[color])
        responses.append(f"Mouth Open:{mouth_count}")
        responses_color.append(color_code[color])
    
    return responses, responses_color,blink_count,mouth_count,prev_eyes,prev_mouth



                
