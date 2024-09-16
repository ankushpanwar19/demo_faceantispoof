import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
from pathlib import Path
import numpy as np
from test import multimodal_antispoof, movement
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
    try:
        while True:
            # Receive video frame from the client
            # print("Received")
            frame = await websocket.receive_bytes()
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
            responses = {"face_detection":"",
                        "face_antispoof":"",
                        "face_antispoof_c":"",
                        "object_detected":"",
                        "object_detected_c":"",
                        "object_antispoof":"",
                        "object_antispoof_c":"",
                        "eyes_movement":"",
                        "mouth_movement":""}
            # try:
            if no_of_faces == 'single_face':
                try:
                    responses,blink_count,mouth_count,prev_eyes,prev_mouth = movement(responses,image_from_array,blink_count,mouth_count,prev_eyes,prev_mouth)
                except: 
                    responses["eyes_movement"] = "Landmark Model failed"
                    responses["mouth_movement"] = "Landmark Model failed"
            elif no_of_faces == 'no_face':
                responses["face_detection"] = "No Face detected"
                responses["face_detection_c"] = "red"
            else :
                responses["face_detection"] = "Multiple Face detected"
                responses["face_detection_c"] = "red"
            # except:
            #     print("Prediction Error")

            # response = { "Face Spoof": "Real","Object Spoof": "Fake", "Detection": "True"}
            # Echo the frame back to the client (for display, if needed)
            await websocket.send_bytes(json.dumps(responses))
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
