import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import IPython.display as ipd


def record():
    samplerate = 16000  
    duration = 1 # seconds
    filename = 'test.wav'
    print("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    print("end")
    sd.wait()
    sf.write(filename, mydata, samplerate)
    return filename
    
def predict(classes, model, audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

def test(classes, model, filename = 'test.wav'):
    samples, sample_rate = librosa.load(filename, sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples,rate=8000)  
    return predict(classes, model, samples)