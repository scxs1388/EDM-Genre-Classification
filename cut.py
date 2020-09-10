import os
import wave
import numpy as np


def cut_audio():
    source_path = r'D:/Programming/musicsample/fakedubstepsample'
    file_list = os.listdir(source_path)
    # print(file_list)
    file_list = [source_path + '/' + fp for fp in file_list]
    cut_duration = 30
    proc = 0
    for file_name in file_list:
        f = wave.open(r'' + file_name, 'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]  # 声道数, 量化位数, 采样频率, 采样点数
        str_data = f.readframes(nframes)
        f.close()

        wave_data = np.frombuffer(str_data, dtype=np.short)
        if nchannels > 1:
            wave_data.shape = -1, 2
            wave_data = wave_data.T
            temp_data = wave_data.T
        else:
            # eave_data = wave_data.T
            temp_data = wave_data.T

        cut_frame_num = framerate * cut_duration
        cut_num = nframes / cut_frame_num
        step_num = int(cut_frame_num)
        # step_total_num = 0

        # for i in range(int(cut_num)):
        #     file_path = r'D:/Programming/musicsample/realdubstepsample' + '/' + file_name.split('/')[-1].split('.wav')[
        #         0] + '(' + str(i) + ')' + '.wav'
        #     temp_data_temp = temp_data[step_num * i: step_num * (i + 1)]
        #     # step_total_num = (i + 1) * step_num
        #     temp_data_temp.shape = 1, -1
        #     temp_data_temp = temp_data_temp.astype(np.short)
        #     f = wave.open(file_path, 'wb')
        #     f.setnchannels(nchannels)
        #     f.setsampwidth(sampwidth)
        #     f.setframerate(framerate)
        #     f.writeframes(temp_data_temp.tostring())  # 将wav_data转换为二进制数据写入文件
        #     f.close()
        # proc += 1
        # print(f'process: {proc}/{len(file_list)}')
        # #
        # for i in range(2, 4):
        #     file_path = r'D:/Programming/musicsample/realdubstepsample' + '/' + file_name.split('/')[-1].split('.wav')[0] + '('+str(i)+')' + '.wav'
        #     temp_data_temp = temp_data[step_num * i: step_num * (i+1)]
        #     step_total_num = (i + 1) * step_num
        #     temp_data_temp.shape = 1, -1
        #     temp_data_temp = temp_data_temp.astype(np.short)
        #     f = wave.open(file_path, 'wb')
        #     f.setnchannels(nchannels)
        #     f.setsampwidth(sampwidth)
        #     f.setframerate(framerate)
        #     f.writeframes(temp_data_temp.tostring())  # 将wav_data转换为二进制数据写入文件
        #     f.close()
        #
        # for i in range(int(cut_num)-3, int(cut_num)-1):
        #     file_path = r'D:/Programming/musicsample/realdubstepsample' + '/' + file_name.split('/')[-1].split('.wav')[0] + '('+str(i)+')' + '.wav'
        #     temp_data_temp = temp_data[step_num * i: step_num * (i+1)]
        #     step_total_num = (i + 1) * step_num
        #     temp_data_temp.shape = 1, -1
        #     temp_data_temp = temp_data_temp.astype(np.short)
        #     f = wave.open(file_path, 'wb')
        #     f.setnchannels(nchannels)
        #     f.setsampwidth(sampwidth)
        #     f.setframerate(framerate)
        #     f.writeframes(temp_data_temp.tostring())  # 将wav_data转换为二进制数据写入文件
        #     f.close()


if __name__ == '__main__':
    cut_audio()
