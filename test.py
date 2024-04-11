import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

actions = ["안녕하세요", "감사합니다", "마음", "휴먼"]
seq_length = 30
model = load_model('models/model_KSL.keras')

# mediapipe 기본 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# 이거는 나중에 이해
seq = []
# 세 개의 액션이 같으면 그 동작을 출력
action_seq = []
this_action = ''


def draw_korean(image, org, text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('fonts/gulim.ttc', 40)
    draw.text(org, text, font=font, fill=(255, 255, 255))
    return np.array(img)


while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 손 랜드마크 감지
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[i for i in range(1, 21)], :3]
            v = v2 - v1  # 3차원에서의 거리 구하기 (벡터)

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

            # 불러온 모델에 데이터를 넣고 저장된 가중치들을 이용해서 값을 예측한다.
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

    frame = draw_korean(frame, (80, 430), this_action)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
