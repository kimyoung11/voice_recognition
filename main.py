import os
import wave
import pylab
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy.signal as signal
import natsort
import librosa
import librosa.display


# -------------------------------------SPECTROGRAM---------------------------------
# spectrogram으로 숫자 예측 계산
def find_spectrogram_similarity(userInput, sample):
    #     print(userInput, sample)
    distArr = []
    if len(userInput) >= len(sample):
        for i in range(0, len(userInput) - len(sample) + 1):
            dist = 0
            for j in range(0, len(sample)):
                dist += distance.euclidean(userInput[i + j], sample[j])
            distArr.append(dist / len(sample))
        #         print(min(distArr))
        return min(distArr)
    else:
        #         print(100)
        return 100


# 해당 숫자 spectrogram 만들기
def graph_spectrogram(wav_file):
    print("spectrogram start")
    sound_info, frame_rate = get_wav_info(wav_file)
    spectrogram, freq, time = mlab.specgram(sound_info, Fs=frame_rate, scale_by_freq=True, sides='default')
    spectrogram = spectrogram_normalize(spectrogram)
    Y = np.log10(np.flipud(spectrogram))
    Z = np.transpose(Y)
    print("spectrogram end")
    #     print(Z)
    #     print('\n\n')
    return Z


# spectrogram에 필요한 정규화
def spectrogram_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# spectrogram을 위한 오디오 전처리
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, dtype=np.uint16)
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


# spectrogram 데이터 셋 정리
def make_data_set(path):
    print("MAKE DATA SET!!!")
    data_set = []
    folder_list = os.listdir(path)
    folder_list = natsort.natsorted(folder_list, reverse=False)
    for folder in folder_list:
        data_set.append(graph_spectrogram(path + '/' + folder))

    return data_set


# spectrogram 정답 도출 매핑 함수
def mapping(test, example, example_list):
    idx = 0
    for example_data, example_name in zip(example, example_list):
        distance = []
        for test_data in test:
            distance.append(find_spectrogram_similarity(example_data, test_data))

        print('ANSWER : ', example_name)
        print('PREDICT : ', distance.index(min(distance))%10)
        print('-----------------------')
        idx = idx + 1


# -------------------------------------FFT------------------------------------
# spectrogram 데이터 셋 정리
def make_fft_data_set(path):
    print("MAKE DATA SET!!!")
    data_set = []
    data_list = []
    folder_list = os.listdir(path)
    folder_list = natsort.natsorted(folder_list, reverse=False)
    for folder in folder_list:
        if folder == '.DS_Store':
            continue
        data_list.append(folder)
        mag, sampling_rate = wav_fft(path + '/' + folder)
        mag_db = librosa.amplitude_to_db(mag)
        mag_n = fft_normalize(mag_db)
        data_set.append(mag_n)

    return data_set,data_list


# 해당 숫자 fft 만들기
def wav_fft(wav_file):
    #     print("fft start")
    audio_sample, sampling_rate = librosa.load(wav_file, sr=None)
    fft_result = librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length=1024, window=signal.hann).T
    mag, phase = librosa.magphase(fft_result)
    #     print("fft end")
    return mag, sampling_rate


# fft에 필요한 정규화
def fft_normalize(s):
    min_level_db = -100
    return np.clip((s - min_level_db) / (-min_level_db), 0, 1)


# fft plt 출력 함수
def print_fft(path):
    idx = 331
    folder_list = os.listdir(path)
    folder_list = natsort.natsorted(folder_list, reverse=False)
    for folder in folder_list:
        if folder == '.DS_Store':
            continue
        mag, sampling_rate = wav_fft(path + '/' + folder)
        mag_db = librosa.amplitude_to_db(mag)
        mag_n = fft_normalize(mag_db)

        librosa.display.specshow(mag_n.T, y_axis='linear', x_axis='time', sr=sampling_rate)
        print(folder)
        print(mag_n)
        print("mag length : ", len(mag_n))
        plt.show()
        idx = idx + 1


if __name__ == '__main__':
    #     print_fft('/Users/Desktop/4-1/패턴인식/VoiceTotal/f1')
    #     print_fft('/Users/Desktop/4-1/패턴인식/VoiceTotal/test1')

    #     #기준
    test_path = './dataset'
    test_set,test_list = make_fft_data_set(test_path)


    #     #예측값
    example_path = './test_soundSize'
    example_set,example_list = make_fft_data_set(example_path)


    mapping(test_set, example_set, example_list)
