import requests
import subprocess
import re
import time
import json
import sounddevice as sd
# import numpy as np
import whisper
import tempfile
import soundfile as sf
import warnings
import numpy as np 
import soundfile as sf
import warnings
import queue 
from listener import HybridSpeechListener
from speaker import Mimic3SentenceStreamer_V2
import os 
import pyttsx3
import torchaudio 
import torch 
import datetime
from pysilero_vad import SileroVoiceActivityDetector
import logging
import pyttsx3
logging.getLogger().setLevel(logging.WARNING)
### Silero VAD Wrapper Class ###
class SileroVADWrapper:
    def __init__(self, input_sample_rate=48000, vad_sample_rate=16000, threshold=0.5):
        self.input_sample_rate = input_sample_rate
        self.vad_sample_rate = vad_sample_rate
        self.threshold = threshold
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
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
class WhisperListener:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.block_duration = 0.50  # 250ms
        self.block_size = int(self.sample_rate * self.block_duration)
        self.recording = []
        self.vad = SileroVADWrapper(input_sample_rate=48000)
        self.whisper_model = whisper.load_model("base") 
        self.stop_flag = False
    def listen_and_transcribe(self):
        # global voice_activity_status
        self.recording = []
        self.stop_flag = False
        silence_counter = 0
        silence_threshold = 6 # 8  
        print("Recording... Speak now.")
        def audio_callback(indata, frames, time_info, status):
            nonlocal silence_counter  
            audio_block = indata[:, 0].copy()
            vad_probs = self.vad.process(audio_block)
            speech_detected = any(prob >= self.vad.threshold for prob in vad_probs)
            if speech_detected:
                print("Speech detected!")
                self.recording.append(audio_block)
                silence_counter = 0
            else:
                print('No Speech is detected ')
                silence_counter += 1
            if silence_counter >= silence_threshold:
                self.stop_flag = True
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                            blocksize=self.block_size, callback=audio_callback):
            while not self.stop_flag:
                time.sleep(0.1)
        if self.recording:
            full_audio = np.concatenate(self.recording)
            with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                sf.write(f.name, full_audio, self.sample_rate)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('always')
                    result = self.whisper_model.transcribe(f.name , language="en")
                    print('Transcribe:', result['text'])
                    if "next question" in result['text'].lower():
                        print("Detected 'next question' keyword.")
                    return result['text']
        else:
            print("No speech detected.")
            return "candidate not give the anwser for this question"
class Mimic3SentenceStreamer:
    def __init__(self, listen_module, voice="en_US/cmu-arctic_low", pause_after_sentence=0.3):
        self.buffer = ""
        self.voice = voice
        self.pause_after_sentence = pause_after_sentence
        self.listen = listen_module  
    def add_token(self, token):
        self.buffer += token
        if re.search(r'[.!?]$', self.buffer.strip()):
            self.flush()
    def flush(self):
        # self.listen.pause()
        sentence = self.buffer.strip()
        if sentence:
            print(f"Speaking: {sentence}")
            subprocess.run(["mimic3", "--voice", self.voice,"--length-scale", "1.2", sentence ])
            time.sleep(self.pause_after_sentence)
            # self.listen.resume()
            time.sleep(0.3)
        self.buffer = ""
    def finish(self):
        if self.buffer.strip():
            self.flush()
    def speak(self, text):
        # self.listen.pause()
        if text:
            subprocess.run(["mimic3", "--voice", self.voice,"--length-scale", "1.4", text])
            time.sleep(self.pause_after_sentence)
        # self.listen.resume()
        time.sleep(0.3)

class Pyttsx3SentenceStreamer:
    def __init__(self, listen_module=None, pause_after_sentence=0.3, rate=120):
        self.buffer = ""
        self.listen = listen_module
        self.pause_after_sentence = pause_after_sentence
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
    def add_token(self, token):
        self.buffer += token
        if re.search(r'[.!?]$', self.buffer.strip()):
            self.flush()
    def flush(self):
        sentence = self.buffer.strip()
        if sentence:
            print(f"Speaking: {sentence}")
            self.engine.say(sentence)
            self.engine.runAndWait()
            time.sleep(self.pause_after_sentence)
        self.buffer = ""
    def finish(self):
        if self.buffer.strip():
            self.flush()
    def speak(self, text):
        if text:
            print(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            time.sleep(self.pause_after_sentence)
class Interviewer:
    def __init__(self,job_role ,  custom_questions=None ):
        self.conversation = ""
        self.question_count = 0
        if custom_questions:
            self.max_questions = len(custom_questions)
        else:
            self.max_questions = 5
        self.current_index = 0
        self.feed_back_prompt  = f"""
You are an expert interview evaluator reviewing a spoken interview transcript for the role of {job_role}. This interview was conducted using speech-to-text (STT) transcription, so please be mindful that some parts of the candidate's spoken answers may not have been transcribed accurately.
Your task is to evaluate the candidate fairly based on the content that was captured, and give feedback that could help them improve for real-world interviews and company referrals.
Your response must include:
1. **Per-Question Feedback**:
    - Restate each question briefly.
    - Provide natural, objective feedback on the candidate's response:
        - What was understood correctly or explained well.
        - What was missing or unclear — indicate whether it could be due to transcription issues.
        - Suggest better or clearer ways the answer could have been framed.
    - Do not use tag formats like "Correct:", "Improvement:", etc. Write in full sentences.
    - Avoid generic praise or repeated phrases.
2. **Also give the overall review based in the candidate answer and which part the candidate wants to improve the performance 
2. **Category-Wise Evaluation** (after all questions):
    Evaluate the candidate in each area with brief comments:
    - **Technical Knowledge**
    - **Communication & Fluency**
    - **Sentence Structure & Clarity**
    - **Confidence & Speaking Pace**
    - **Resilience to STT Misunderstanding** (i.e., how understandable their speech was despite minor transcription errors)
3. **Overall Rating**:
    - Provide a **score out of 10**, reflecting their knowledge, articulation, and clarity.
    - Give the socore correctly based on the performance and correctly 
    - Give  overly generous and not overly critical.
4. **Final Summary**:
    - Summarize the candidate’s overall suitability for the {job_role} position.
    - Mention whether they are ready for real-world interviews or need more practice.
    - Phrase this professionally for sharing with companies or mentors.
Instructions:
- If you notice incomplete or unclear answers, assume minor STT errors may be responsible and mention it politely.
- Keep the tone professional, constructive, and encouraging.
- Avoid long explanations or repetition.
"""     
        self.feed_back_prompt = f"""
You are an expert technical interviewer evaluating a candidate for the position of **{job_role}**. The candidate's answers were transcribed using speech-to-text (STT), so minor errors may appear in the transcript. Be honest and fair, but do not assume the candidate said something unless it's clearly present.

Evaluate the candidate based on their **actual responses**. If they skipped a question or gave an incomplete answer, clearly state it.

Your response must include the following sections:

---

**1. Per-Question Feedback**
- For each question, restate it briefly.
- Clearly state if the candidate answered it or not.
- If answered:
  - Mention what was explained well.
  - Point out missing/incorrect parts (do NOT assume correctness if it wasn't stated).
  - Be brief and to the point.
- If skipped or unclear, say: _"Candidate did not answer this question clearly or skipped it."_

---

**2. Category-Wise Ratings (Score 0–10)**
Rate each category with a short comment (1–2 lines max):
- **Technical Knowledge**:
- **Communication**:
- **Clarity & Structure**:
- **Confidence**:
- **Handling of STT Errors**:

---

**3. Overall Evaluation**
- One paragraph: Is the candidate ready for real interviews? What should they improve?
- Give a final score out of 10.

---

**Instructions**
- Do NOT invent or assume correct answers.
- Be strict but supportive.
- If a question wasn't answered, do NOT give credit.
- Keep your feedback **professional and concise**.
"""

        # self.feed_back_prompt = feedback_prompt 
        self.custom_questions = custom_questions or []
        self.questions_history = []
        self.current_question = ""
        # self.listen =  WhisperListener()
        self.listen = HybridSpeechListener()
        # self.listen.start()
        self.tts_streamer = Mimic3SentenceStreamer_V2(listen_module=self.listen)
        # self.tts_streamer = Pyttsx3SentenceStreamer()
        self.is_paused = False 
        self.stopped = False 
        self.job_role  =  job_role
        self.feedback = ''
        self.active = False 
    def ask_custom_question(self, question):
        self.current_question = question 
        self.questions_history.append(question)
        if question != "Interview Finished":
            self.tts_streamer.speak(question )
            self.conversation += f"\nInterviewer: {question}"
            
    
        
        return True 
    def ask_next_question(self):
        if self.current_index < len(self.questions):
            self.current_question = self.questions[self.current_index]
            self.questions_history.append(self.current_question)
            self.current_index += 1
        else:
            self.current_question = "Interview Finished"
            self.active = False 
    def ask_ai_question(self):
        payload = {
            'model': 'llama3:8b',
            'prompt': self.system_prompt + self.conversation + "\nInterviewer:",
            'stream': True
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        result = ''
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8')
                data = json.loads(chunk)
                token = data.get('response', '')
                result += token
                if "INTERVIEW FINISHED THANK YOU" in token:
                    continue
                self.tts_streamer.add_token(token)
        self.tts_streamer.flush()
        self.conversation += f"\nInterviewer: {result}"
        return True 
    def get_feedback(self):
        print('Generating Feedback....')
        payload = {
            'model': 'llama3:8b',
            'prompt': self.feed_back_prompt + self.conversation,
            'stream': True
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        result = ''
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8')
                data = json.loads(chunk)
                token = data.get('response', '')
                result += token
                self.tts_streamer.add_token(token)
        self.tts_streamer.flush()
        return result
    def get_feedback_v2(self):
        start_time =  time.time()

        print('Generating Feedback....')
        payload = {
            'model': 'llama3:8b',
            'prompt': self.feed_back_prompt + self.conversation,
            'stream': True
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        result = ''
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8')
                data = json.loads(chunk)
                token = data.get('response', '')
                result += token
        # self.tts_streamer.flush()
        end_time =time.time()
        print(f"Feed back Time : {end_time - start_time}")
        return result
    def speak_feedback(self):
        
        payload = {
            'model': 'llama3:8b',
            'prompt': self.feed_back_prompt + self.conversation,
            'stream': True
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        result = ""
        for line in response.iter_lines():  
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        chunk = data["response"]
                        result += chunk
                except json.JSONDecodeError as e:
                    print("JSON decode error:", e)
                    continue
                result
        self.tts_streamer.speak()
        return result
    def get_answer(self):
        answer = ""
        retries = 5  
        while not answer.strip() and retries > 0:
            print("\nListening for your answer...")
            answer = self.listen.get_transcript()
            if answer.strip():
                # self.conversation += f"\nCandidate: {answer}"
                return answer
            retries -= 1
        print("No valid answer detected. Moving on.")
        return "(No valid answer)"
        
    def conduct_interview(self):
        while self.question_count < self.max_questions:
            if self.stopped:
                print('Interview Stopped')
                return 
            while self.is_paused:
                print("Interview Paused")
                time.sleep(1)
            if self.question_count < len(self.custom_questions):
                question = self.custom_questions[self.question_count]
                ques_flag = self.ask_custom_question(question)
            # else:
            #     ques_flag  = self.ask_ai_question()
            # candidate_text = input("Candidate: ")
            while self.is_paused:
                print("Interview Paused")
                time.sleep(1)
            if ques_flag:
                time.sleep(0.7)
                if self.stopped:
                    print('Interview Stopped')
                    return
                candidate_text  = self.listen.listen_and_transcribe()
                print('Transcribe :' , candidate_text)
            
                self.conversation += f"\nCandidate: {candidate_text}"
                self.question_count += 1
        start_time = time.time()
        convo_file = f"{self.job_role}_interview_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filedir = 'interviewRecord'
        os.makedirs(filedir, exist_ok=True)
        filepath =  os.path.join(filedir , convo_file)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.conversation)
        print(f"Conversation saved to {filepath}")
        # self.feedback = self.speak_feedback()
        # print("Generating feedback...")
        # self.feedback = self.speak_feedback()
        # print(self.feedback)
        end_time =  time.time()
        print('Overall time :{}'.format(end_time - start_time))
        if 'INTERVIEW FINISHED THANK YOU' in self.feedback:
            print("Interview finished!")
        print('Interview Finished ')
    def pause(self):
        self.is_paused = True  
    def stop(self):
        self.stopped =  True 
        self.current_question = "Interview Finished"
    def resume(self):
        self.is_paused  =  False 
    def reset(self):
        self.conversation = ""
        self.question_count = 0
        self.questions_history = []
        self.stopped = False
        self.feedback = ""
if __name__ == '__main__':
        start_time_overall =  time.time()
        my_custom_questions = [
            "Explain what happens when you type 'LS' in Linux.",
            "How do you check memory usage on a Linux server?",
            "Interview Finished"
        ]
        agent = Interviewer(custom_questions=my_custom_questions , job_role='linux')
        agent.conduct_interview()
        end_time_overall =  time.time()
        print('Overall interview time :{}'.format(end_time_overall-start_time_overall))