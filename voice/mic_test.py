import time
import sounddevice as sd, soundfile as sf
from faster_whisper import WhisperModel

SR = 16000
model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)

input("Press Enter, then speak for 5 seconds...")
audio = sd.rec(int(5 * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
sf.write("mic_test.wav", audio, SR)

print("Playing back...")
sd.play(audio, SR); sd.wait()

t = time.time()
segments, _ = model.transcribe(audio[:, 0], language="en", beam_size=1)
text = " ".join(s.text.strip() for s in segments)
print(f"Transcript: {text}")
print(f"Transcription took {time.time() - t:.2f} s")
