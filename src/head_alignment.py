import os
import math
import cv2
import numpy as np
from src.utils import calcAngle
from scipy.fft import fft2, fftshift

def check_lighting(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        average_intensity = np.mean(gray)
    
        # Define a threshold for face lighting conditions (adjust as needed)
        threshold = 90.0
        
        # Check if the average intensity exceeds the threshold
        is_good_lighting = average_intensity > threshold
        lighting_percentage = average_intensity
    except:
        return False, 0.0
    return is_good_lighting, lighting_percentage

def lighting_check(face_frame):
    is_good_lighting, lighting_percentage = check_lighting(face_frame)
    out = ['good' if is_good_lighting else 'bad', round(lighting_percentage,2)]

    return out

def find_major_landmark(face_landmarks,image_width,image_height):
    # left EYE print
    pt1 = int(face_landmarks.landmark[33].x * image_width), int(face_landmarks.landmark[33].y * image_height)
    pt2 = int(face_landmarks.landmark[133].x * image_width), int(face_landmarks.landmark[133].y * image_height)
    left_eye = int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2)
    

    # right EYE print
    pt1 = int(face_landmarks.landmark[362].x * image_width), int(face_landmarks.landmark[362].y * image_height)
    pt2 = int(face_landmarks.landmark[263].x * image_width), int(face_landmarks.landmark[263].y * image_height)
    right_eye = int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2)
    

    # nose
    nose = int(face_landmarks.landmark[1].x * image_width), int(face_landmarks.landmark[1].y * image_height)
    
    # lip side 57,287
    left_lip_corner = int(face_landmarks.landmark[62].x * image_width), int(face_landmarks.landmark[62].y * image_height)
    right_lip_corner = int(face_landmarks.landmark[292].x * image_width), int(face_landmarks.landmark[292].y * image_height)
    center_up_lip = int(face_landmarks.landmark[13].x * image_width), int(face_landmarks.landmark[13].y * image_height)
    center_down_lip = int(face_landmarks.landmark[14].x * image_width), int(face_landmarks.landmark[14].y * image_height)
    center_lip = int((center_up_lip[0]+center_down_lip[0])/2),int((center_up_lip[1]+center_down_lip[1])/2)

    # check depth of forhead and chin
    head = int(face_landmarks.landmark[10].x * image_width), int(face_landmarks.landmark[10].y * image_height)
    chin = int(face_landmarks.landmark[199].x * image_width), int(face_landmarks.landmark[199].y * image_height)
    depth_diff_head_chin = face_landmarks.landmark[9].z - face_landmarks.landmark[199].z
    return left_eye, right_eye, nose, left_lip_corner, right_lip_corner, center_lip, head, chin, depth_diff_head_chin


def check_alignment2(face_landmarks,rgb_frame):

    image_height,image_width = rgb_frame.shape[0], rgb_frame.shape[1]

    #check head tilt
    left_eye, right_eye, nose, left_lip_corner, right_lip_corner, center_lip, head, chin, depth_diff_head_chin = find_major_landmark(face_landmarks,image_width,image_height)

    z_tilt = calcAngle(left_eye,right_eye,(0,0),(1,0))

    # check y_side_facing
    left_eye_angle = 180+calcAngle(nose,left_eye,left_eye,right_eye)
    right_eye_angle = -1*calcAngle(nose,right_eye,left_eye,right_eye)
    y_side_look = right_eye_angle-left_eye_angle

    # # check x_up_facing
    x_uplook = calcAngle(left_lip_corner, right_lip_corner,left_lip_corner, center_lip)
    # right_eye_angle = -1*calculate_angle(nose,right_eye)
    # y_side_look = right_eye_angle-left_eye_angle

    return z_tilt , y_side_look , x_uplook,depth_diff_head_chin

def check_head_alignment(face_landmarks,rgb_frame):
    z_tilt , y_side_look , x_uplook,depth_diff_head_chin  = check_alignment2(face_landmarks,rgb_frame)

    head_tilt = ['tilted' if abs(z_tilt)>20 else 'straight', round(z_tilt,2)]

    head_left_right = ['side' if abs(y_side_look)>25 else 'straight', round(y_side_look,2)]

    print("depth_diff_head_chin: ",depth_diff_head_chin)
    if depth_diff_head_chin<-0.05:
        head_up_down ='looking_down'
        # head_up_down ='look_straight'
    elif depth_diff_head_chin>0.05:
        head_up_down ='looking_up'
        # head_up_down ='look_straight'
    else:
        head_up_down = 'straight'
    head_up_down = [head_up_down, round(depth_diff_head_chin,2)]

    return head_tilt, head_left_right, head_up_down


def check_blurr(rgb_image):
    
    try:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        (h, w,) = gray.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        size = 10
        
        fftShift = fftshift(fft2(gray))
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        # Calculate the magnitude spectrum
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        
        # Calculate the mean magnitude in the high-frequency region
    
        
        # Normalize the mean magnitude
        normalized_mean_magnitude = mean
        
        # Define a threshold for blur detection (adjust as needed)
        threshold = 20
        
        # Check if the normalized mean magnitude in the high-frequency region is above the threshold
        is_blurr = normalized_mean_magnitude < threshold
        lap_var = normalized_mean_magnitude
    except:
        return False, 0.0

    
    return is_blurr, lap_var
            