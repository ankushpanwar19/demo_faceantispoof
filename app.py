from fastapi import FastAPI, WebSocket
import cv2
from src.face_detection import detect_face, detect_landmarks
from test import multimodal_antispoof,movement
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/test")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    blink_count = 0
    prev_eyes = ''
    mouth_count = 0
    prev_mouth = ''
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_box, no_of_faces = detect_face(rgb_frame)
        if no_of_faces == 'single_face':
            cv2.putText(frame, "Face: Single Face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            responses, responses_color = multimodal_antispoof(rgb_frame,face_box)
            t = 0 
            for i, resp in enumerate(responses):
                cv2.putText(frame, resp, (10, 20+(i+1)*35), cv2.FONT_HERSHEY_SIMPLEX, 1,responses_color[i] , 2)
                t=i

            responses, responses_color,blink_count,mouth_count,prev_eyes,prev_mouth = movement(rgb_frame,blink_count,mouth_count,prev_eyes,prev_mouth)
            for i, resp in enumerate(responses):
                t=t+1
                cv2.putText(frame, resp, (10, 20+(t+1)*35), cv2.FONT_HERSHEY_SIMPLEX, 1,responses_color[i] , 2)
                
            
        elif no_of_faces == 'no_face':
            cv2.putText(frame, "Face: No Face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face: Multiple Face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process frame (face detection, etc.)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Send frame to the client
        await websocket.send_bytes(frame_bytes)

    cap.release()