import warnings

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from scipy import fft
from scipy.io import wavfile
warnings.filterwarnings('ignore')

# 音频文件根目录
sample_path = ''
# 音频文件类0别
genres = []
# 每种类别音频文件的目录
music_paths = [sample_path + g for g in genres]
# 每个音频文件的路径
music_file_paths = []
for mp in music_paths:
    music_file_paths.append([mp + '/' + music_file for music_file in os.listdir(mp)])

import time
data_set = []
target_set = []
genre2id = {g: i for i, g in enumerate(genres)}
id2genre = {i: g for i, g in enumerate(genres)}
sample = music_file_paths[0][0]
start = time.clock()
genre_size = 200
for i, g in enumerate(genres):
    process = 0
    for file_name in music_file_paths[i][:genre_size]:
        y, sr = librosa.load(file_name, mono=True)
        # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        S, phase = librosa.magphase(librosa.stft(y))
        rmse = librosa.feature.rmse(S=S)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=512)

        # Compute global onset autocorrelation
        ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
        ac_global = librosa.util.normalize(ac_global)

        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_rolloff)} {np.mean(zcr)} {np.mean(ac_global)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        data_set.append([float(i) for i in to_append.split(' ')])
        target_set.append(genre2id[g])
        process += 1
        print(f'genre:{g}\tprocess:{process}/{genre_size}')

end = time.clock()
print(f'Wall Time: {end - start} seconds.')

# 固定随机种子，打乱数据集
state = np.random.get_state()
np.random.shuffle(data_set)
np.random.set_state(state)
np.random.shuffle(target_set)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data_set, dtype=float))
y = utils.to_categorical(np.array(target_set))

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建神经网络模型
model = models.Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.003)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.2)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=70, batch_size=128)

# 预测结果
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

