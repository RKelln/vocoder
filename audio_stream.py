import pyaudio
import numpy as np
import threading

import webrtcvad
import librosa

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHUNK_SIZE = 2048


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10**(m / 2595) - 1)


class VoiceActivityDetector:
    def __init__(self, mode=1, # Modes 0-3, higher is more aggressive
                 frame_duration:int=30, # ms
                 resample_rate:int=16000):  # Hz
        self.vad = webrtcvad.Vad(mode)  
        self.frame_duration = frame_duration
        self.resample_rate = resample_rate
        self.speaking = False


    def is_speech(self, audio_chunk, sr=DEFAULT_SAMPLE_RATE):
        # Normalize audio data to [-1.0, 1.0]
        audio_float32 = audio_chunk.astype(np.float32) / 32768.0

        # Resample audio to 16kHz if necessary
        if sr != self.resample_rate:
            audio_float32 = librosa.resample(audio_float32, orig_sr=sr, target_sr=self.resample_rate)
            sr = self.resample_rate

        # Convert back to 16-bit PCM
        audio_int16 = np.clip(audio_float32 * 32768, -32768, 32767).astype(np.int16)

        # WebRTC VAD requires 10, 20, or 30 ms frames
        frame_length = int(sr * self.frame_duration / 1000)

        # Ensure the chunk is the right length
        # if len(audio_int16) < frame_length:
        #     # Pad with zeros if too short
        #     audio_int16 = np.pad(audio_int16, (0, frame_length - len(audio_int16)), 'constant')
        # elif len(audio_int16) > frame_length:
        #     # Trim the audio to the required frame length
        #     audio_int16 = audio_int16[:frame_length]

        frame_length = int(sr * self.frame_duration / 1000)
        audio_length = len(audio_int16)
        num_frames = audio_length // frame_length

        is_speech_detected = False

        # Perform VAD on each frame
        for i in range(num_frames):
            frame = audio_int16[i * frame_length : (i + 1) * frame_length]
            audio_bytes = frame.tobytes()
            if self.vad.is_speech(audio_bytes, sample_rate=sr):
                is_speech_detected = True
                break

        #is_speech = self.vad.is_speech(audio_bytes, sample_rate=sr)
        if self.speaking != is_speech_detected:
            self.speaking = is_speech_detected
            #print("Speaking:", is_speech_detected, num_frames)
        return is_speech_detected


class AudioStream:
    def __init__(self, rate=DEFAULT_SAMPLE_RATE, chunk_size=DEFAULT_CHUNK_SIZE):
        self.rate = rate
        self.chunk_size = chunk_size
        self.running = False

    def start_stream(self):
        self.running = True

    def get_audio_chunk(self):
        raise NotImplementedError("Subclasses must implement this method")

    def stop_stream(self):
        self.running = False
        
    def __enter__(self):
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_stream()

    def __del__(self):
        if self.running:
            self.stop_stream()

    def sample_rate(self):
        return self.rate

class PyAudioStream(AudioStream):
    def __init__(self, rate=DEFAULT_SAMPLE_RATE, chunk_size=DEFAULT_CHUNK_SIZE):
        super().__init__(rate, chunk_size)
        self.stream = None
        self.audio_interface = pyaudio.PyAudio()
        self.buffer = []
        self.lock = threading.Lock()

    def start_stream(self):
        super().start_stream()

    def get_audio_chunk(self):
        raise NotImplementedError("Subclasses must implement this method")

    def stop_stream(self):
        super().stop_stream()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio_interface.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        with self.lock:
            self.buffer.append(data)
        return (None, pyaudio.paContinue)

    def get_audio_chunk(self):
        with self.lock:
            if self.buffer:
                return self.buffer.pop(0)
            else:
                return None

    def stop_stream(self):
        super().stop_stream()
        

class PyAudioFileAudioStream(PyAudioStream):
    def __init__(self, file_path, rate=DEFAULT_SAMPLE_RATE, chunk_size=DEFAULT_CHUNK_SIZE, loop=False):
        super().__init__(rate, chunk_size)
        self.file_path = file_path
        self.loop = loop
        self.audio_data = self.load_audio_file(file_path)
        self.current_position = 0

    def start_stream(self):
        super().start_stream()
        self.current_position = 0
        # Open a PyAudio output stream with a callback
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback,
        )
        self.stream.start_stream()

    def load_audio_file(self, file_path):
        # Load audio file at its original sample rate
        audio_data, original_rate = librosa.load(file_path, sr=None, mono=True)
        # Resample if the sample rates don't match
        if original_rate != self.rate:
            audio_data = librosa.resample(audio_data, orig_sr=original_rate, target_sr=self.rate)
        # Clip the audio data to prevent overflow/underflow
        audio_data = np.clip(audio_data, -1.0, 1.0)
        # Convert to int16 format
        audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
        return audio_data

    def callback(self, in_data, frame_count, time_info, status):
        if not self.running:
            return (None, pyaudio.paComplete)

        until_end = len(self.audio_data) - self.current_position
        if frame_count > until_end:
            if self.loop:
                remainder = frame_count - until_end
                chunk = np.concatenate((self.audio_data[self.current_position:], self.audio_data[:remainder]))
                self.current_position = remainder % len(self.audio_data)
            else:
                chunk = self.audio_data[self.current_position:]
                # Pad the rest with zeros if not looping
                chunk = np.pad(chunk, (0, frame_count - until_end), mode='constant')
                self.current_position = len(self.audio_data)
                self.running = False
                # Append the last chunk to the buffer
                with self.lock:
                    self.buffer.append(chunk)
                return (chunk.tobytes(), pyaudio.paComplete)
        else:
            chunk = self.audio_data[self.current_position : self.current_position + frame_count]
            self.current_position += frame_count
            if self.current_position >= len(self.audio_data) and not self.loop:
                self.running = False

        # Append chunk to buffer
        with self.lock:
            self.buffer.append(chunk)

        return (chunk.tobytes(), pyaudio.paContinue)


class MicAudioStream(PyAudioStream):

    def start_stream(self):
        super().start_stream()
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback
        )
