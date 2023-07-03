import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import pipeline
import pyaudio
import numpy as np
import openai
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import config

openai.api_key = config.OPENAI_API_KEY

# in seconds
RECORDING_LENGTH = 5

# ORIGINAL WHISPER SMALL CHECKPOINT
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# FINE-TUNED WHISPER SMALL CHECKPOINT
model = AutoModelForSpeechSeq2Seq.from_pretrained("tince250/whisper-small-sr-ri", revision="5001970fc5aef30ed4bf44b81de19ed74ce9bf3d")
processor = AutoProcessor.from_pretrained("tince250/whisper-small-sr-ri")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="serbian", task="transcribe")

# pipe = pipeline(model="tince250/whisper-small-sr-ri")

def speak(text):
    tts = gTTS(text=text, lang='sr')
    tts.save("output.mp3")

    mixer.init()
    mixer.music.load("output.mp3")
    mixer.music.play()

    while mixer.music.get_busy():
        continue
    mixer.quit()


# def transcribe_audio(audio):
#     text = pipe(audio)["text"]
#     return text

def transcribe_audio(audio):
    sample = audio
    input_features = processor(sample, sampling_rate=16000, return_tensors="pt").input_features 

    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

def record_audio(seconds):
    chunk = 1024
    sample_format = pyaudio.paFloat32
    channels = 1
    fs = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []
    for _ in range(int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.concatenate([np.frombuffer(frame, dtype=np.float32) for frame in frames], axis=0)
    return audio

def main():
    speak("Kako mogu pomoći?")

    while True:
        print("Speak:")

        audio = record_audio(RECORDING_LENGTH)  

        print("Transcribing...")
        transcription = transcribe_audio(audio)

        print("Transcription:", transcription)

        # transcription = "Koliko protona ima u atomu vodonika?"

        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content":
                                "You are a helpful assistant that speaks Serbian."},
                                {"role": "user", "content": transcription},
                            ],
                            temperature=0.5,
                            max_tokens=200,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            n=1,
                            stop=["\nUser:"],
                        )

        bot_response = response["choices"][0]["message"]["content"]

        print("JARVIS RESPONSE:")
        print(bot_response)
        speak(bot_response)

if __name__=="__main__":
    main()