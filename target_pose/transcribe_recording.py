import numpy as np
import sounddevice as sd
import os
import speech_recognition as sr
from scipy.io.wavfile import write
import wavio
import azure.cognitiveservices.speech as speechsdk


# r = sr.Recognizer()

# fs = 44100
# path = os.path.dirname(os.path.realpath(__file__)) + '/training_data1'
# myarray = np.load(path + '/recordings/recording_0003.npy')
# print(myarray)
# print(myarray.shape)
# sd.play(myarray, fs)
# sd.wait()







def transcribe_azure(speech_config,counter,path):

    myarray = np.load(path + '/recordings/recording_{}.npy'.format(str(counter).zfill(4)))
    wavio.write(path + '/recordings/recording_{}.wav'.format(str(counter).zfill(4)), myarray, 44100 ,sampwidth=1)

    audio_input = speechsdk.audio.AudioConfig(filename=path + '/recordings/recording_{}.wav'.format(str(counter).zfill(4)))

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = speech_recognizer.recognize_once()
    print(result.text)
    return result.text


def main():
    speech_key, service_region = '4203927a90be4b1785cee6bdd8310f48', "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    path = os.path.dirname(os.path.realpath(__file__)) + '/training_data1'



    transcribed = open(path + '/data_transcribed_new.csv', 'a')
    with open('{}/data.csv'.format(path), 'r' ) as file:
        for i,line in enumerate(file):
            if i > 105:
                transcribed_text = transcribe_azure(speech_config,i,path).replace(',', '').replace('.', '')
                line = line.strip() + transcribed_text + ',' + transcribed_text + '\n'
                transcribed.write(line)
    transcribed.close()

main()

        





