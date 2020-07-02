import numpy as np
import sounddevice as sd
import os
import speech_recognition as sr
from scipy.io.wavfile import write
import wavio

r = sr.Recognizer()

fs = 44100
path = os.path.dirname(os.path.realpath(__file__)) + '/training_data'
myarray = np.load(path + '/recordings/recording_0014.npy')
print(myarray)
print(myarray.shape)
sd.play(myarray, fs)
sd.wait()


wavio.write(path + '/recordings/recording_0014.wav', myarray, fs ,sampwidth=1)


harvard = sr.AudioFile(path + '/recordings/recording_0014.wav',)
with harvard as source:
    audio = r.record(source)

print(type(audio))
print(r.recognize_google(audio,language='en-IN',show_all = True))



