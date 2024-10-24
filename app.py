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
from test import multimodal_antispoof, movement, face_oval, head_alignment, env_check, face_blur_check,perspective_distortion_ratio
from src.face_detection import detect_face

app = FastAPI()

# Enable CORS for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0

    blink_count = 0
    prev_eyes = ''
    mouth_count = 0
    prev_mouth = ''
    consecutive_capture = 0
    objectspoof = []
    facespoof = []
    try:
        while True:
            # Receive video frame from the client
            # print("Received")
            # frame = await websocket.receive_bytes()
            print("********** Next ***********")
            # print("yessss")
            data = await websocket.receive_text()
            json_data = json.loads(data)

            # Extract the image and array from the received JSON
            image_base64 = json_data['image']
            ovalCoords = json_data['ovalCoords']
            oval_enlarge = json_data['ovalEnlarge']
            print(oval_enlarge)
            ovalCoords = list(ovalCoords)
            ovalCoords = [ float(num.split('p')[0]) for num in ovalCoords]

            # Decode the Base64 image
            frame = base64.b64decode(image_base64.split(',')[1])  
            # print(type(frame))
            if len(frame) == 0:
                print("Received empty data, skipping...")
                continue
            # Convert bytes to an image using Pillow
            image = Image.open(io.BytesIO(frame))

            # Convert the image to RGB format
            rgb_image = image.convert("RGB")
            image_array = np.array(rgb_image)
            # print(image_array.shape)

            # PREDICTION
            face_box, no_of_faces = detect_face(image_array)
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
                        "final_face_spoof":"",
                        "perspective_ratio":"",
                        }
            # try:
            if no_of_faces == 'single_face':
                is_good_lighting, is_blurr = env_check(image_array)
                head_tilt=['straight',0]
                head_left_right=['straight',0]
                head_up_down=['straight',0]
                try:
                    
                    head_tilt, head_left_right, head_up_down = head_alignment(image_array)
                except:
                    print("Landmark Model ERROR")
                    responses["eyes_movement"] = "Landmark model failed"
                    responses["mouth_movement"] = "Landmark model failed"

                try: 
                    face_oval_check, bottom_check, area_check,area_percent = face_oval(image_array,ovalCoords)
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
                elif face_oval_check:
                    # print("yess")
                    responses = multimodal_antispoof(image_array,face_box)

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
                                
                        perspective_ratio = perspective_distortion_ratio(image_array)
                        responses['perspective_ratio'] = perspective_ratio
                    
                else:
                    if area_percent>100:
                        responses["face_detection"] = "Distance yourself from the camera!"
                    elif area_percent<100 and (not area_check):
                        responses["face_detection"] = "Come closer to the camera!"                    
                    elif not bottom_check:
                        responses["face_detection"] = "Align your chin with oval bottom!"
                    else:
                        responses["face_detection"] = "Position your face completely inside the oval!"
                    responses["oval_alignment"] = False
                    consecutive_capture = 0
                    objectspoof = []
                    facespoof = []
                
                if (head_tilt[0]=='straight' or head_left_right[0]=='straight' or head_up_down[0]=='straight'):
                    try:
                        responses,blink_count,mouth_count,prev_eyes,prev_mouth = movement(responses,image_array,blink_count,mouth_count,prev_eyes,prev_mouth)
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
            await websocket.send_bytes(json.dumps(responses))
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
