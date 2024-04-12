import os
import cv2
import time
import numpy as np
import mediapipe as mp

actions = ["hello", "thanks", "sorry", "hate", "mind", "person", "thinking", "friend", "school"]  # 원하는 동작 설정
seq_length = 9
secs_for_action = 30  # 최대 학습 시간

# mediapipe에 있는 hands 모델
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 2로 바꿀 예정
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

file = 'dataset_KSL'
os.makedirs(file, exist_ok=True)  # if 문이 필요가 없음

while cap.isOpened():
    for idx, action in enumerate(actions):
        ret, frame = cap.read()
        data = []

        frame = cv2.flip(frame, 1)  # 이건 왜 있는지 잘 모름
        cv2.putText(
            frame,
            text=f'Waiting for collecting {action.upper()} action',
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )
        cv2.imshow('frame', frame)
        cv2.waitKey(6000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, frame = cap.read()

            # frame 작업
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 이거 공부
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.x, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[i for i in range(1, 21)], :3]
                    v = v2 - v1  # 20행, 3열

                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 열 추가

                    angle = np.arccos(
                        np.einsum(
                            'nt,nt->n',
                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                        )
                    )

                    angle = np.degrees(angle)

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d)

                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        np.save(os.path.join(file, f'raw_{action}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])
        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join(file, f'seq_{action}'), full_seq_data)
    break
