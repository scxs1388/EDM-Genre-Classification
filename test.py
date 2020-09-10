import warnings

import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import tensorflow as tf
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from scipy import fft
from scipy.io import wavfile
warnings.filterwarnings('ignore')
process = 0
genre_size = 3000
# genres = ['ambient', 'dubstep', 'metal', 'piano', 'raptrap', 'trance']
genres = ['ambient', 'trance']

# for i, g in enumerate(genres):g
#     process = 0
#     for file_name in music_file_paths[i][:genre_size]:
#         y, sr = librosa.load(file_name, mono=True, offset=60, duration=30)
#         # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         S, phase = librosa.magphase(librosa.stft(y))
#         rmse = librosa.feature.rmse(S=S)
#
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#
#         oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
#         tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=512)
#
#         # Compute global onset autocorrelation
#         ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
#         ac_global = librosa.util.normalize(ac_global)
#
#         to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_rolloff)} {np.mean(zcr)} {np.mean(ac_global)}'
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         data_set.append([float(i) for i in to_append.split(' ')])
#         target_set.append(genre2id[g])
#         process += 1
#         print(f'genre:{g}\tprocess:{process}/{genre_size} name:{file_name}')


def feature_extract(file_name):
    y, sr = librosa.load(file_name, mono=True, offset=0, duration=30)
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
    ac_global = librosa.util.normalize(librosa.autocorrelate(oenv, max_size=tempogram.shape[0]))
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_rolloff)} {np.mean(zcr)} {np.mean(ac_global)}'
    # print(len(mfcc), len(mfcc[0]))
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    ds = ([float(i) for i in to_append.split(' ')])
    global process
    process += 8
    print(('process: %.3f%%' % (100*process/len(genres)/genre_size))+f'|{process}/{len(genres)*genre_size}|{file_name}')
    return ds


def read_genre_file(genre):
    return np.load(f'data/{genre}_data_set.npy')


if __name__ == '__main__':
    # 音频文件根目录
    sample_path = 'D:/Programming/musicsample/'
    # 音频文件的6种类别
    # genres = ['drumnbass', 'dubstep', 'hardstyle', 'prohouse', 'trance', 'trap']
    # genres = ['trance', 'piano']
    # 每种类别音频文件的目录
    music_paths = [sample_path + 'real' + g + 'sample' for g in genres]
    # 每个音频文件的路径
    music_file_paths = []
    for mp in music_paths:
        music_file_paths.append([mp + '/' + music_file for music_file in os.listdir(mp)])

    data_set = []
    target_set = []
    genre2id = {g: i for i, g in enumerate(genres)}
    id2genre = {i: g for i, g in enumerate(genres)}
    start = time.clock()

    for i, g in enumerate(genres):
        with ProcessPoolExecutor(max_workers=8) as executor:
            music_file_list = music_file_paths[i][:genre_size]
            da_se = list(executor.map(feature_extract, music_file_list))
        data_set.extend(da_se)
        target_set.extend([i]*genre_size)
        np.save(f'{g}_data_set.npy', np.array(da_se))

    for i, g in enumerate(genres):
        target_set.extend([i] * genre_size)
        data_set.extend(read_genre_file(g))

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
    print(X)
    print(y)

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 创建神经网络模型
    model = models.Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.003)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.003)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.003)))
    model.add(Dropout(0.5))
    # model.add(Dense(6, activation='softmax'))
    model.add(Dense(len(genres), activation='softmax'))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=512)

    # 预测结果
    test_acc = []
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    test_acc.append(test_accuracy)
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')

    r = 0
    for i, t in enumerate(X_train):
        # print(t)
        pred = model.predict_classes(np.array([t]))
        print(pred, np.argmax(y_train[i]))
        if pred[0] == np.argmax(y_train[i]):
            r += 1
    print(f'================\n{r/len(X_train)}')
    modesave = tf.train.Saver()
    saver
    model.save('mgc_model.h5')


