import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")

def label(path):
    labels=os.listdir(path)

    #find count of each label and plot bar graph
    no_of_recordings=[]
    for label in labels:
        waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))

    #plot
    plt.figure(figsize=(30,5))
    index = np.arange(len(labels))
    plt.bar(index, no_of_recordings)
    plt.xlabel('Commands', fontsize=12)
    plt.ylabel('No of recordings', fontsize=12)
    plt.xticks(index, labels, fontsize=15, rotation=60)
    plt.title('No. of recordings for each command')
    plt.show()
    labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    return labels
    
def wave_form(path):
    samples, sample_rate = librosa.load(path, sr = 16000)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + str(path))
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
    
def sound(path):
    samples, sample_rate = librosa.load(path, sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    return ipd.Audio(samples, rate=8000)

def model_perf(history):
    plt.plot(history.history['loss'], label='train') 
    plt.plot(history.history['val_loss'], label='test') 
    plt.legend() 
    plt.show()