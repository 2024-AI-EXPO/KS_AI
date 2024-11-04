import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Flatten
from keras.initializers import Orthogonal
from keras.models import save_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

path = "./dataset"
actions = [label.split('.')[0] for label in os.listdir(path)]
data = np.concatenate([np.load(path + f'/{action}.npy') for action in actions], axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]
y_data = to_categorical(labels, num_classes=len(actions))
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

initializers = Orthogonal(gain=1.0, seed=None)
dr = 0.3

model = Sequential()
model.add(Input(x_data.shape[1:]))

model.add(LSTM(128, return_sequences=True, kernel_initializer=initializers))
model.add(Flatten())

model.add(Dense(64, activation='relu', kernel_initializer=initializers))
model.add(Dropout(dr))
model.add(Dense(len(actions), activation='softmax', kernel_initializer=initializers))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    x_data,
    y_data,
    epochs=125,
    batch_size=16,
)
save_model(model, "./models/symbols-v2-2.keras")

fig, loss_ax = plt.subplots(figsize=(12, 5))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train_loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val_loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(history.history['accuracy'], 'b', label='train_acc')
# acc_ax.plot(history.history['val_accuracy'], 'g', label='val_acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='center right')
plt.ylim(0.0, 1.1)
plt.show()
