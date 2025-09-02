import requests
import sounddevice as sd
import soundfile as sf
import io
import time
import re

class Mimic3SentenceStreamer_V2:
    def __init__(self, listen_module=None, voice="en_US/cmu-arctic_low", 
                 pause_after_sentence=0.3, server_url="http://localhost:59125"):
        self.buffer = ""
        self.voice = voice
        self.pause_after_sentence = pause_after_sentence
        self.listen = listen_module
        self.server_url = server_url
        self.normal_length_scale = 1.2
        self.speak_length_scale = 1.4
        
    def _synthesize_and_play(self, text, length_scale):
        """Synthesize text to speech using Mimic3 HTTP API and play it"""
        params = {
            'voice': self.voice,
            'lengthScale': str(length_scale),
            'noiseScale': "0.667",
            'noiseW': "0.8",
        }
        
        try:
            # Send text to Mimic3 server
            response = requests.post(
                f"{self.server_url}/api/tts",
                params=params,
                data=text.encode('utf-8'),
                headers={'Content-Type': 'text/plain'},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Mimic3 API error: {response.status_code} - {response.text}")
                return
                
            # Create in-memory WAV file
            with io.BytesIO(response.content) as wav_io:
                # Read audio data
                data, fs = sf.read(wav_io)
                
                # Play audio
                sd.play(data, fs)
                sd.wait()  # Wait until playback is finished
                
        except Exception as e:
            print(f"Error in Mimic3 synthesis: {e}")

    def add_token(self, token):
        self.buffer += token
        if re.search(r'[.!?]$', self.buffer.strip()):
            self.flush()

    def flush(self):
        sentence = self.buffer.strip()
        if sentence:
            print(f"Speaking: {sentence}")
            self._synthesize_and_play(sentence, self.normal_length_scale)
            time.sleep(self.pause_after_sentence)
        self.buffer = ""

    def finish(self):
        if self.buffer.strip():
            self.flush()

    def speak(self, text):
        if text:
            print(f"Speaking: {text}")
            self._synthesize_and_play(text, self.speak_length_scale)
            time.sleep(self.pause_after_sentence)# Initialize the streamer



if __name__ == "__main__":            
    tts_streamer = Mimic3SentenceStreamer()

    # Speak a sentence
    tts_streamer.speak("Hello, this is a cross-platform text-to-speech system!")

    # Stream tokens
    tts_streamer.add_token("This is ")
    tts_streamer.add_token("a streamed ")
    tts_streamer.add_token("sentence.")
    tts_streamer.finish()