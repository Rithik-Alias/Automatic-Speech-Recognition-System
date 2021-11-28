import librosa
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def remove_shorter_waves(labels, path):
    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr = 16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if(len(samples)== 8000) : 
                all_wave.append(samples)
                all_label.append(label)

    all_wave = np.array(all_wave).reshape(-1,8000,1)
    return all_wave, all_label

def preprocess(all_label, all_wave, labels):
    le = LabelEncoder()
    y=le.fit_transform(all_label)
    classes= list(le.classes_)
    y=np_utils.to_categorical(y, num_classes=len(labels))
    all_wave = np.array(all_wave).reshape(-1,8000,1)
    return train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True), classes
