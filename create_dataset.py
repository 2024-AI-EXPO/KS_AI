import os
import cv2
import time
import numpy as np
import mediapipe as mp

# actions = ["hello", "thanks", "sorry", "hate", "hungry",
#            "sick", "tired", "mind", "person", "think",
#            "friend", "school", "police", "rice", "bed"]
save_path = "final_dataset"
os.makedirs(save_path, exist_ok=True)

action = "tough"
seq_length = 5
secs_for_action = 15
idx = 16

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

start_time = time.time()
data = []
while time.time() - start_time < secs_for_action and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.x, lm.visibility]

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

            angle_label = np.array([angle], dtype=np.float32)
            angle_label = np.append(angle_label, idx)

            d = np.concatenate([joint.flatten(), angle_label])
            data.append(d)

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow(action, frame)
    if cv2.waitKey(10) == ord('q'):
        break

data = np.array(data)

full_seq_data = []
for seq in range(len(data) - seq_length):
    full_seq_data.append(data[seq:seq + seq_length])
full_seq_data = np.array(full_seq_data)
print(action, full_seq_data.shape)
np.save(os.path.join(save_path, f'seq_{action}'), full_seq_data)

cap.release()
cv2.destroyAllWindows()
