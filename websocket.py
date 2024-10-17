import cv2
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

actions = ["안녕하세요", "감사합니다", "미안합니다", "싫어합니다", "배고프다",
           "아프다", "졸리다", "마음", "사람", "생각", "친구", "학교", "경찰", "쌀밥", "침대"]

seq_length = 5

model = load_model('models/KSL1.keras')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    model_complexity=0
)

def load_font(font_path, font_size):
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        return ImageFont.load_default()

font = load_font('gulim.ttc', 50)

def draw_korean(image, org, text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text(org, text, font=font, fill=(255, 255, 255))
    return np.array(img)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    seq = []
    action_seq = []
    this_action = ''
    buf = ''
    police = 0
    last_process_time = time.time()

    try:
        while True:
            data = await websocket.receive_bytes()

            # current_time = time.time()
            # if current_time - last_process_time < 0.05:
            #     continue
            # last_process_time = current_time

            image = Image.open(BytesIO(data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                    angle = np.degrees(angle)

                    d = np.concatenate([joint.flatten(), angle])
                    seq.append(d)

                    if len(seq) < seq_length:
                        continue

                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                    y_pred = model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.8:
                        continue

                    action = actions[i_pred]
                    action_seq.append(action)

                    if len(action_seq) < 4:
                        continue

                    if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4]:
                        this_action = action
                    else:
                        this_action = '?'

                    if buf == '경찰' and this_action == '사람':
                        this_action = '경찰관'

                    if this_action == "경찰":
                        buf = this_action

            frame = draw_korean(frame, (40, 200), this_action)
            if this_action == '경찰관':
                police += 1
                if police >= 35:
                    buf = ''
                    police = 0

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_bytes = buffer.tobytes()

            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        cv2.destroyAllWindows()
