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
from test import multimodal_antispoof, movement, face_oval, head_alignment, env_check
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

            # print("yessss")
            data = await websocket.receive_text()
            json_data = json.loads(data)

            # Extract the image and array from the received JSON
            image_base64 = json_data['image']
            ovalCoords = json_data['ovalCoords']
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
            image_from_array = np.array(rgb_image)
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
                    face_oval_check = face_oval(image_from_array,ovalCoords)
                except:
                    print("Landmark Face oval Error")
                    face_oval_check = False
                    consecutive_capture = 0

                if not is_good_lighting:
                    responses["face_detection"] = "Please be in a well-lit environment"
                    consecutive_capture = 0
                elif is_blurr:
                    responses["face_detection"] = "Video feed is blurry"
                    consecutive_capture = 0
                elif face_oval_check:
                    # print("yess")
                    responses = multimodal_antispoof(image_from_array,face_box)

                    if (head_tilt[0]!='straight' or head_left_right[0]!='straight' or head_up_down[0]!='straight'):
                        responses["face_detection"] = "Keep your head aligned and straight."
                        consecutive_capture = 0
                        responses["oval_alignment"] = True
                    else:
                        responses["face_detection"] = "Perfect! Stay Still"
                        responses["face_detection_c"] = "green"
                        responses["oval_alignment"] = True
                        consecutive_capture +=1
                        objectspoof.append(responses["object_antispoof"]=='Spoof')
                        facespoof.append(responses["face_antispoof"]=='Spoof')
                        print("consecutive_capture:",consecutive_capture)
                        if consecutive_capture > 5:
                            print("Spoofcheck",facespoof)
                            responses["final_object_spoof"] = "Spoof" if sum(objectspoof)>2 else "Real"
                            responses["final_face_spoof"] = "Spoof" if sum(facespoof)>2 else "Real"
                            # consecutive_capture = 0
                            print("final_object_spoof: ",responses["final_object_spoof"])
                            print("final_face_spoof: ",responses["final_face_spoof"])
                        
                else:
                    responses["face_detection"] = "Align your face within the oval and fill it"
                    responses["oval_alignment"] = False
                    consecutive_capture = 0
                
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
            else :
                responses["face_detection"] = "Multiple Face detected"
                responses["face_detection_c"] = "red"
                blink_count = 0
                prev_eyes = ''
                mouth_count = 0
                prev_mouth = ''
                consecutive_capture = 0
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
