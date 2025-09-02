import os, time, tempfile, warnings
import torch, whisper, torchaudio, sounddevice as sd
import numpy as np
import soundfile as sf
from vosk import Model as VoskModel, KaldiRecognizer
import noisereduce  as nr 

### Silero VAD ###
class SileroVADWrapper:
    def __init__(self, input_sample_rate=48000, vad_sample_rate=16000, threshold=0.5):
        self.input_sample_rate = input_sample_rate
        self.vad_sample_rate = vad_sample_rate
        self.threshold = threshold
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        self.resampler = torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=vad_sample_rate)
        self.buffer = torch.empty(0)

    def process(self, audio_chunk):
        audio_tensor = torch.from_numpy(audio_chunk).float()
        resampled = self.resampler(audio_tensor)
        self.buffer = torch.cat([self.buffer, resampled])
        vad_results = []
        while self.buffer.shape[0] >= 512:
            frame = self.buffer[:512]
            self.buffer = self.buffer[512:]
            with torch.no_grad():
                speech_prob = self.model(frame, self.vad_sample_rate).item()
                vad_results.append(speech_prob)
        return vad_results


### Hybrid Listener ###
class HybridSpeechListener:
    def __init__(self, sample_rate=48000, vosk_model_path="vosk-model-small-en-us-0.15"):
        self.sample_rate = sample_rate
        self.block_duration = 0.5
        self.block_size = int(self.sample_rate * self.block_duration)
        self.recording = []
        self.vad = SileroVADWrapper(input_sample_rate=sample_rate)
        self.whisper_model = whisper.load_model("base")
        self.vosk_model = VoskModel(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        self.stop_flag = False

    def listen_and_transcribe(self):
        self.recording = []
        self.stop_flag = False
        silence_counter = 0
        silence_threshold = 6

        print("Listening for speech...")

        def callback(indata, frames, time_info, status):
            nonlocal silence_counter
            audio_block = indata[:, 0].copy()
            vad_probs = self.vad.process(audio_block)
            speech_detected = any(prob >= self.vad.threshold for prob in vad_probs)

            if speech_detected:
                print("Speech detected.")
                self.recording.append(audio_block)
                silence_counter = 0
            else:
                silence_counter += 1
                print("Silence...")

            if silence_counter >= silence_threshold:
                self.stop_flag = True

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                            blocksize=self.block_size, callback=callback):
            while not self.stop_flag:
                time.sleep(0.1)

        if self.recording:
            full_audio = np.concatenate(self.recording)

            # full_audio =  torch.from_numpy(full_audio)
            # full_audio =  self.resampler(full_audio)
            wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(wav_path, full_audio, self.sample_rate)

            try:
                self.recognizer.Reset()
                with sf.SoundFile(wav_path, 'r') as f:
                    while True:
                        data = f.read(4000)
                        vosk_data =  f.buffer_read(4000 , dtype = 'int16')
                        if len(data) == 0:
                            break
                        self.recognizer.AcceptWaveform(data)

                vosk_result = self.recognizer.FinalResult()
                if '"text" : "' in vosk_result:
                    import json
                    text = json.loads(vosk_result)["text"].strip()
                    if text:
                        print("Vosk Transcription:", text)
                        return text
            except Exception as e:
                print("Vosk failed:", e)

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = self.whisper_model.transcribe(wav_path, language='en')
                print(" Whisper Transcription:", result['text'])
                return result['text']

        else:
            print("No speech detected.")
            return "No speech was detected."
        


if __name__ == "__main__":
    listener = HybridSpeechListener()
    text = listener.listen_and_transcribe()

    if "next question" in text.lower():
        print("➡️ Triggering next question logic...")
