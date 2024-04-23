import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

actions = ["안녕하세요", "감사합니다", "미안합니다", "싫어합니다", "배고프다",
           "아프다", "졸리다", "마음", "사람", "생각",
           "친구", "학교", "경찰", "쌀밥", "침대"]
seq_length = 5
model = load_model('/Users/yabbi/Desktop/GitHub/KS_AI/models/test.keras')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)


def draw_korean(image, org, text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/Users/yabbi/Desktop/GitHub/KS_AI/gulim.ttc', 40)
    draw.text(org, text, font=font, fill=(255, 255, 255))
    return np.array(img)


def generate_frames():
    seq = []
    action_seq = []
    this_action = ''
    cap = cv2.VideoCapture(0)
    buf = ''
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[i for i in range(1, 21)], :3]
                v = v2 - v1

                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(
                    np.einsum(
                        'nt,nt->n',
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                    )
                )

                angle = np.degrees(angle)

                d = np.concatenate([joint.flatten(), angle])
                seq.append(d)

                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                print(conf)
                if conf < 0.8:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 4:
                    continue

                if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4]:
                    this_action = action
                    action_seq = []
                    if this_action == '사람' and buf == '경찰':
                        this_action = '경찰관'

                    if this_action == "경찰":
                        buf = this_action

        frame = draw_korean(frame, (80, 430), this_action)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.get("/AI")
async def stream_frames(background_tasks: BackgroundTasks):
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == '__main__':
    uvicorn.run(app ="test:app",
                host="0.0.0.0",
                port=5955,
                reload=False,
                workers=1)