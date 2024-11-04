import os
import cv2
import time
import numpy as np
import mediapipe as mp


save_path = "dataset"
os.makedirs(save_path, exist_ok=True)

actions = ["hello", "thanks", "sorry", "you", "i",
           "we", "im_ok", "im_full", "hungry", "add_in",
           "crying", "sad", "very_good", "admit", "sick",
           "cold", "tough", "understand", "tired", "awkward",
           "easy", "hate", "waste", "do_it", "happy"]
idx = 8
action = actions[idx]
seq_length = 5
secs_for_action = 30


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=3,
    min_tracking_confidence=0.6,
    min_detection_confidence=0.6,
    model_complexity=1,
)

cap = cv2.VideoCapture(0)

start_time = time.time()
data = []
while time.time() - start_time < secs_for_action and cap.isOpened():
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
                joint[j] = [lm.x, lm.y, lm.x, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[i for i in range(1, 21)], :3]
            v = v2 - v1
            vector_size = np.linalg.norm(v, axis=1)
            v = v / vector_size[:, np.newaxis]

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

            if handed.classification[0].label == 'Left':
                l_vector.append(vector_size)
                l_hl.append(res)
                ld.append(d)
            else:
                r_vector.append(vector_size)
                r_hl.append(res)
                rd.append(d)

        if l_vector:
            freq_left = np.argmax(np.bincount(np.argmax(np.array(l_vector), axis=0)))
            mp_drawing.draw_landmarks(frame, l_hl[freq_left], mp_hands.HAND_CONNECTIONS)
            data.append(ld[freq_left])

        if r_vector:
            freq_right = np.argmax(np.bincount(np.argmax(np.array(r_vector), axis=0)))
            mp_drawing.draw_landmarks(frame, r_hl[freq_right], mp_hands.HAND_CONNECTIONS)
            data.append(rd[freq_right])

    cv2.imshow(action, frame)
    if cv2.waitKey(10) == ord('q'):
        break

data = np.array(data)

seq_data = []
for seq in range(len(data) - seq_length):
    seq_data.append(data[seq:seq + seq_length])
seq_data = np.array(seq_data)
print(action, seq_data.shape)
np.save(os.path.join(save_path, f'seq_{action}1'), seq_data)

cap.release()
cv2.destroyAllWindows()
