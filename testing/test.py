import cv2
import time
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model

actions = ["안녕하세요", "감사합니다", "죄송합니다", "너", "나",
           "우리", "괜찮습니다", "배부르다", "배고프다", "넣다",
           "울다", "슬프다", "훌륭하다", "인정", "아프다",
           "감기", "힘들다", "알아듣다", "졸리다", "어색하다",
           "쉽다", "싫어하다", "아깝다", "할 수 있다", "행복"]
seq_length = 5
model = load_model('../models/symbols-v2-2.keras')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=3,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

def draw_korean(image, org, text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('gulim.ttc', 40)
    draw.text(org, text, font=font, fill=(0, 0, 0))
    return np.array(img)


def generate_frames():
    seq = []
    text = " "
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame = cv2.resize(frame, (w*2, h*2))
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks is not None:
            ld, rd = [], []
            l_hl, r_hl = [], []
            l_vector, r_vector = [], []
            for res, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[i for i in range(1, 21)], :3]
                v = v2 - v1
                vector_size = np.linalg.norm(v, axis=1)  # 벡터의 크기
                v = v / vector_size[:, np.newaxis]
                angle = np.arccos(
                    np.einsum(
                        'nt,nt->n',
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                    )
                )
                angle = np.degrees(angle)

                if handed.classification[0].label == 'Left':
                    l_vector.append(vector_size)
                    l_hl.append(res)
                    ld.append(np.concatenate([joint.flatten(), angle]))
                else:
                    r_vector.append(vector_size)
                    r_hl.append(res)
                    rd.append(np.concatenate([joint.flatten(), angle]))

            if l_vector:
                l_vector = np.array(l_vector)
                freq_left = np.argmax(np.bincount(np.argmax(l_vector, axis=0)))
                mp_drawing.draw_landmarks(frame, l_hl[freq_left], mp_hands.HAND_CONNECTIONS)
                seq.append(ld[freq_left])
            if r_vector:
                r_vector = np.array(r_vector)
                freq_right = np.argmax(np.bincount(np.argmax(r_vector, axis=0)))
                mp_drawing.draw_landmarks(frame, r_hl[freq_right], mp_hands.HAND_CONNECTIONS)
                seq.append(rd[freq_right])

            if len(seq) >= seq_length:
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                action = actions[i_pred]
                print(conf, action)
                if conf >= 0.9:
                    text = action

        frame = draw_korean(frame, (150, 240), text)
        cv2.imshow('', frame)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_frames()
