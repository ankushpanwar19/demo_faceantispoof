import math
from src.face_detection import detect_landmarks
from src.face_antispoof import predict_facespoof
from src.light_antispoof import light_predict_facespoof
from src.object_detection import run_detector, load_objdetection_model
from src.perspective_distortion import perspective_transformation
from src.face_movements import eye_blink,mouth_movement,eye_blink_ear
from src.head_alignment import check_lighting,check_blurr,check_head_alignment


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
    color_code = {"Green":(0, 255, 0), "Red":(0,0,255)}
    responses = {}
    if FACE_ANTISPOOF:
        facespoof_thres = 0.15
        res = light_predict_facespoof(rgb_frame,face_bbox)
        out = "Spoof" if res>facespoof_thres else "Real"
        # res = predict_facespoof(rgb_frame,face_bbox)
        # out = "Real" if res>facespoof_thres else "Spoof"
        print("Face Result:",res)
        res = round(res,3)
        color = "green" if out=='Real' else "red"
        responses["face_antispoof"]=out
        responses["face_antispoof_c"]=color
        print(responses["face_antispoof"])
    
    if OBJECT_DETECTION:
        obj_spoof = "Real"
        obj_detected, obj_bbox = run_detector(rgb_frame)
        responses["object_detected"]=obj_detected
        responses["object_detected_c"]='green'
        for i, obj in enumerate(obj_detected):
            if obj in ['laptop','tv','cellphone','book','remote']:
                ymin,xmin,ymax,xmax = obj_bbox[i]
                margin = 20
                a = bbox_x >= max(0,xmin-margin)
                b = bbox_y >= max(0,ymin-margin)
                c = min(image_width, bbox_x + bbox_width) <= min(image_width, xmax+margin)
                d = min(image_height, bbox_y + bbox_height) <= min(image_height, ymax+margin)
                if a and b and c and d:
                    obj_spoof = "Spoof"
                    break
                
        out = obj_spoof
        color = "green" if out=='Real' else "red"
        responses["object_antispoof"]= out
        responses["object_antispoof_c"]=color
    
    

    return responses


def movement(responses,rgb_frame,blink_count,mouth_count,prev_eyes,prev_mouth):

    landmarks,face_landmarks = detect_landmarks(rgb_frame)
    # blink_count, prev_eyes = eye_blink(landmarks,blink_count,prev_eyes)
    blink_count, prev_eyes = eye_blink_ear(landmarks,blink_count,prev_eyes)
    mouth_count, prev_mouth = mouth_movement(landmarks,mouth_count,prev_mouth)

    # print(blink_count, prev_eyes)
    color = "Green"
    responses["eyes_movement"]= blink_count
    responses["mouth_movement"]= mouth_count
    
    return responses,blink_count,mouth_count,prev_eyes,prev_mouth

head_top = [109,10,338]
head_bottom = [148,152,377]
head_left = [93,234,127]
head_right = [323,454,356]
head_map = [head_top,head_bottom,head_left,head_right]
center = [5]

def face_oval(rgb_frame,oval_coords):
    """
    oval_coords : (cx,cy,rx,ry)
    rgb_frame : numpy array 
    cx = width/2 and cy = height/2 (ideally)
    """
    landmark,face_landmarks = detect_landmarks(rgb_frame)
    # print("Frame shape: ",rgb_frame.shape)
    # print("oval_coords:",oval_coords)
    frame_y = rgb_frame.shape[0]
    frame_x = rgb_frame.shape[1]
    cx,cy,rx,ry = oval_coords
    factor_x = (frame_x/2)/cx
    factor_y = (frame_y/2)/cy
    cx,cy,rx,ry = cx*factor_x,cy*factor_y,rx*factor_x,ry*factor_y
    # print("new_oval_coords:",(cx,cy,rx,ry))
    oval_top = cy-ry
    oval_bottom = cy+ry
    oval_left = cx-rx
    oval_right = cx+rx
    oval_area = math.pi*rx*ry
    def margin_percentage(a,b):
        # if a<b:
        #     return 0.5
        return (a-b)/b
    # top_check = True
    # print("centerx",(landmark[center[0]][0]-cx)/cx)
    # print("centery",(landmark[center[0]][1]-cy)/cy)
    try:
        center_check = abs((landmark[center[0]][0]-cx)/cx)<0.2 and abs((landmark[center[0]][1]-cy)/cy)<0.3
        thres = -0.05
        top_check = margin_percentage(landmark[head_map[0][0]][1],oval_top)>thres or margin_percentage(landmark[head_map[0][1]][1],oval_top)>thres or margin_percentage(landmark[head_map[0][2]][1],oval_top)>thres

        bottom_check = margin_percentage(oval_bottom, landmark[head_map[1][0]][1])>thres or margin_percentage(oval_bottom, landmark[head_map[1][1]][1])>thres or margin_percentage(oval_bottom, landmark[head_map[1][2]][1])>thres
        bottom_check2 = margin_percentage(oval_bottom, landmark[head_map[1][0]][1])<0.15 or margin_percentage(oval_bottom, landmark[head_map[1][1]][1])<0.15  or margin_percentage(oval_bottom, landmark[head_map[1][2]][1])<0.15
        bottom_check =  bottom_check and bottom_check2
        left_check = margin_percentage(landmark[head_map[2][0]][0],oval_left)>thres or margin_percentage(landmark[head_map[2][1]][0],oval_left)>thres or margin_percentage(landmark[head_map[2][2]][0],oval_left)>thres

        right_check = margin_percentage(oval_right, landmark[head_map[3][0]][0])>thres or margin_percentage(oval_right,landmark[head_map[3][1]][0])>thres or margin_percentage(oval_right, landmark[head_map[3][2]][0])>thres

        face_ry = (landmark[head_map[1][1]][1] - landmark[head_map[0][1]][1])/2
        face_rx = (landmark[head_map[3][1]][0] - landmark[head_map[2][1]][0])/2
        face_area = math.pi*face_rx*face_ry
        area_percent = face_area/oval_area*100
        area_check = True if area_percent<100 and area_percent>40 else False
        print(round(area_percent,2),center_check, top_check, bottom_check, left_check, right_check)
        # check = (top_check and bottom_check and left_check and right_check) or (top_check and bottom_check and left_check and right_check) or (top_check and bottom_check and left_check and right_check)
        return area_check and center_check and (top_check and bottom_check and left_check and right_check)
    except:
        return False
    

def env_check(rgb_image):
    is_good_lighting, lighting_percentage = check_lighting(rgb_image)

    # print(is_good_lighting, lighting_percentage)

    is_blurr, lap_var = check_blurr(rgb_image)
    
    return is_good_lighting, is_blurr

def head_alignment(rgb_frame):
    landmark,face_landmarks = detect_landmarks(rgb_frame)
    head_tilt, head_left_right, head_up_down = check_head_alignment(face_landmarks,rgb_frame)

    return head_tilt, head_left_right, head_up_down

