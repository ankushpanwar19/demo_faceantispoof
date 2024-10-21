import os
import base64
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
from pathlib import Path
import numpy as np
from test import multimodal_antispoof, movement, face_oval, head_alignment, env_check, face_blur_check
from src.face_detection import detect_face
import cv2
import base64
from io import BytesIO
from PIL import Image


frame_count = 0

blink_count = 0
prev_eyes = ''
mouth_count = 0
prev_mouth = ''
consecutive_capture = 0
objectspoof = []
facespoof = []

def websocket_endpoint(frame,ovalCoords):
    
    try:
        print("********** Next ***********")

        # print(image_from_array.shape)

        # PREDICTION
        face_box, no_of_faces = detect_face(image_from_array)
        responses = {"Process":"processing",
                    "face_detection":"",
                    "face_antispoof":"",
                    "face_antispoof_c":"",
                    "object_detected":"",
                    "object_detected_c":"",
                    "object_antispoof":"",
                    "object_antispoof_c":"",
                    "eyes_movement":"",
                    "mouth_movement":"",
                    "oval_alignment":False,
                    "final_object_spoof":"",
                    "final_face_spoof":""}
        # try:
        if no_of_faces == 'single_face':
            is_good_lighting, is_blurr = env_check(image_from_array)
            head_tilt=['straight',0]
            head_left_right=['straight',0]
            head_up_down=['straight',0]
            try:
                
                head_tilt, head_left_right, head_up_down = head_alignment(image_from_array)
            except:
                print("Landmark Model ERROR")
                responses["eyes_movement"] = "Landmark model failed"
                responses["mouth_movement"] = "Landmark model failed"

            try: 
                face_oval_check, bottom_check, area_check = face_oval(image_from_array,ovalCoords)
            except:
                print("Landmark Face oval Error")
                face_oval_check = False
                consecutive_capture = 0
                objectspoof = []
                facespoof = []
                bottom_check = False
                area_check = False

            if not is_good_lighting:
                responses["face_detection"] = "Please be in a well-lit environment"
                consecutive_capture = 0
                objectspoof = []
                facespoof = []
            elif is_blurr:
                responses["face_detection"] = "Video feed is blurry"
                consecutive_capture = 0
                objectspoof = []
                facespoof = []
            elif True:
                # print("yess")
                responses = multimodal_antispoof(image_from_array,face_box)

                if (head_tilt[0]!='straight' or head_left_right[0]!='straight' or head_up_down[0]!='straight'):
                    msg=''
                    if head_tilt[0]!='straight':
                        msg = "Head is tilted"
                    elif head_left_right[0]!='straight':
                        msg = "Looking sideways "
                    elif head_up_down[0]!='straight':
                        msg = 'Looking Down'if head_up_down[0]=='looking_down' else 'Looking up' 
                    responses["face_detection"] = f"Keep you head straight (Issue:{msg})"
                    consecutive_capture = 0
                    objectspoof = []
                    facespoof = []
                    responses["oval_alignment"] = False
                else:
                    responses["face_detection"] = "Perfect! Stay Still"
                    responses["face_detection_c"] = "green"
                    responses["oval_alignment"] = True
                    consecutive_capture +=1
                    objectspoof.append(responses["object_antispoof"]=='Spoof')
                    facespoof.append(responses["face_antispoof"]=='Spoof')
                    print("consecutive_capture:",consecutive_capture)
                    if consecutive_capture > 5:
                        print("ObjectSpoof",objectspoof)
                        print("FaceSpoof",facespoof)
                        # print("Spoofcount",sum(facespoof))
                        
                        responses["final_object_spoof"] = "Spoof" if sum(objectspoof)/len(objectspoof)>0 else "Real"
                        responses["final_face_spoof"] = "Spoof" if sum(facespoof)/len(facespoof)>0.2 else "Real"
                        # consecutive_capture = 0
                        print("final_object_spoof: ",responses["final_object_spoof"])
                        print("final_face_spoof: ",responses["final_face_spoof"])
                    
            else:
                if not bottom_check:
                    responses["face_detection"] = "Align your chin with oval bottom!"
                elif not area_check:
                    responses["face_detection"] = "Come closer to the camera!"
                else:
                    responses["face_detection"] = "Align your face within the oval and fill it!"
                responses["oval_alignment"] = False
                consecutive_capture = 0
                objectspoof = []
                facespoof = []
            
            if (head_tilt[0]=='straight' or head_left_right[0]=='straight' or head_up_down[0]=='straight'):
                try:
                    responses,blink_count,mouth_count,prev_eyes,prev_mouth = movement(responses,image_from_array,blink_count,mouth_count,prev_eyes,prev_mouth)
                except:
                    print("Landmark Model ERROR 2")
                    responses["eyes_movement"] = "Landmark model failed"
                    responses["mouth_movement"] = "Landmark model failed"
                    blink_count = 0
                    prev_eyes = ''
                    mouth_count = 0
                    prev_mouth = ''

        elif no_of_faces == 'no_face':
            responses["face_detection"] = "No Face detected"
            responses["face_detection_c"] = "red"
            blink_count = 0
            prev_eyes = ''
            mouth_count = 0
            prev_mouth = ''
            consecutive_capture = 0
            objectspoof = []
            facespoof = []
            
        else :
            responses["face_detection"] = "Multiple Face detected"
            responses["face_detection_c"] = "red"
            blink_count = 0
            prev_eyes = ''
            mouth_count = 0
            prev_mouth = ''
            consecutive_capture = 0
            objectspoof = []
            facespoof = []
        # except:
        #     print("Prediction Error")

        # response = { "Face Spoof": "Real","Object Spoof": "Fake", "Detection": "True"}
        # Echo the frame back to the client (for display, if needed)
        # print(responses)
    except:
        print("issues")

    return responses


def video_to_base64(video_path, frame_interval=30):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames_base64 = []
    frame_count = 0

    while video_capture.isOpened():
        success, frame = video_capture.read()

        # Break the loop if the video has ended
        if not success:
            break

        # Extract frames at the specified interval
        if frame_count % frame_interval == 0:
            # Convert the frame (numpy array) to an image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Save the image in a buffer
            buffered = BytesIO()
            image.save(buffered, format="JPEG")

            # Convert image buffer to base64
            frame_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            frames_base64.append(frame_base64)

        frame_count += 1

    video_capture.release()
    return frames_base64

def base64_to_image(image_base64):

    # if len(frame) == 0:
    #     print("Received empty data, skipping...")
    #     return image_from_array,ovalCoords
    # Convert bytes to an image using Pillow
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(bytes(image_bytes)))

    # Convert the image to RGB format
    rgb_image = image.convert("RGB")
    image_from_array = np.array(rgb_image)

    ovalCoords = [image_from_array.shape[0]/2,image_from_array.shape[1]/2,image_from_array.shape[0]/2,image_from_array.shape[1]/2]

    return image_from_array,ovalCoords
    

if __name__ == "__main__":
    # Example usage
    video_path = "Snapchat baby filter.mov"
    frames_as_base64 = video_to_base64(video_path, frame_interval=30)

    # To print or store the base64 strings
    with open("logfile.txt", "a") as log_file:
    
        for i,frame in enumerate(frames_as_base64):
            image_from_array,ovalCoords = base64_to_image(frame)
            resp = websocket_endpoint(image_from_array,ovalCoords)
            out = str(i)+":"+resp['face_antispoof']+","+resp['object_antispoof']+"\n"
            # print(i,":",resp['face_antispoof'],resp['object_antispoof'])
            log_file.write(out)
