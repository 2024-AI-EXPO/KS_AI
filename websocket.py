# app.py
import cv2
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from io import BytesIO
from PIL import Image , ImageDraw ,ImageFont
import mediapipe as mp
app = FastAPI()

# 액션 리스트 정의
actions = ["안녕하세요", "감사합니다", "미안합니다", "싫어합니다", "배고프다",
           "아프다", "졸리다", "마음", "사람", "생각", "친구", "학교", "경찰", "쌀밥", "침대"]

# 시퀀스 길이
seq_length = 5

# 학습된 모델 로드
model = load_model('/Users/yabbi/Desktop/GitHub/KS_AI/models/KSL1.keras')

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# 한글 텍스트를 영상에 그리는 함수
def draw_korean(image, org, text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/Users/yabbi/Desktop/GitHub/KS_AI/gulim.ttc', 100)
    draw.text(org, text, font=font, fill=(0, 0, 0))
    return np.array(img)


# HTML 클라이언트 페이지
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebSocket Video Stream</title>
</head>
<body>
    <h1>WebSocket Video Stream</h1>
    <video id="video" autoplay playsinline style="width: 640px; height: 480px;"></video>
    <script>
        let video = document.getElementById('video');
        let ws = new WebSocket('ws://localhost:8000/ws');

        // 카메라 스트림 요청 및 오류 처리
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    // 카메라 프레임 전송
                    let canvas = document.createElement('canvas');
                    let ctx = canvas.getContext('2d');

                    setInterval(() => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        canvas.toBlob(blob => {
                            if (ws.readyState === WebSocket.OPEN) {
                                ws.send(blob);
                            }
                        }, 'image/jpeg');
                    }, 100);
                };
            })
            .catch(error => {
                console.error("카메라 접근 실패:", error);
                alert("카메라 접근에 실패했습니다. 브라우저 설정을 확인하세요.");
            });

        ws.onmessage = function(event) {
            let reader = new FileReader();
            reader.readAsDataURL(event.data);
            reader.onloadend = function() {
                let img = new Image();
                img.src = reader.result;
                img.onload = function() {
                    video.srcObject = null;
                    video.src = img.src;
                };
            };
        };

        ws.onclose = function() {
            console.log('WebSocket closed');
        };
    </script>
</body>
</html>
"""

# HTML 페이지 반환
@app.get("/")
async def get():
    return HTMLResponse(html)

# WebSocket 연결 처리
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    seq = []
    action_seq = []
    this_action = ''
    buf = ''
    police = 0

    try:
        while True:
            # 바이트 데이터 수신
            data = await websocket.receive_bytes()

            # 바이트 데이터를 이미지로 변환
            image = Image.open(BytesIO(data))
            frame = np.array(image)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 손 랜드마크 처리
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
                        action_seq = []
                        if buf == '경찰' and this_action == '사람':
                            this_action = '경찰관'

                        if this_action == "경찰":
                            buf = this_action

            frame = draw_korean(frame, (80, 430), this_action)
            if this_action == '경찰관':
                police += 1
                if police >= 35:
                    buf = ''
                    police = 0

            # 프레임을 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # WebSocket을 통해 프레임 전송
            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
