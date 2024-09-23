import pyaudio
import numpy as np
import threading
import queue
import asyncio
import time
from io import BytesIO

import webrtcvad
import librosa
from pydub import AudioSegment
import sounddevice

from hume_stream import hume_stream, DEFAULT_HUME_CONFIG_ID

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHUNK_SIZE = 1024


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
    channels:int
    rate:int
    chunk_size:int
    running:bool

    def __init__(self, channels:int=1, rate:int=DEFAULT_SAMPLE_RATE, chunk_size:int=DEFAULT_CHUNK_SIZE):
        self.channels = channels
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
    
    def get_chunk_duration(self) -> float:  # seconds
        # e.g., 1600 / 16000 = 0.1 sec
        return float(self.chunk_size) / float(self.rate)

class PyAudioStream(AudioStream):
    def __init__(self, device:int=-1, channels:int=1, rate:int=DEFAULT_SAMPLE_RATE, chunk_size:int=DEFAULT_CHUNK_SIZE):
        super().__init__(channels, rate, chunk_size)
        self.stream = None
        self.audio_interface = pyaudio.PyAudio()
        self.device = device
        self.channels = channels
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
        #print(f"Callback: {len(in_data)} bytes, frames: {frame_count}")
        data = np.frombuffer(in_data, dtype=np.int16)
        if not np.any(data):
            print(f"Callback: silence")
        with self.lock:
            self.buffer.append(data)
        return (None, pyaudio.paContinue)

    def get_audio_chunk(self):
        with self.lock:
            if self.buffer:
                return self.buffer.pop(0)
            else:
                return None


class PyAudioFileAudioStream(PyAudioStream):
    def __init__(self, file_path, channels:int=1, rate=DEFAULT_SAMPLE_RATE, chunk_size=DEFAULT_CHUNK_SIZE, loop=False):
        super().__init__(channels, rate, chunk_size)
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
            channels=self.channels,
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

    def __init__(self, device:int=-1, channels:int=None, rate:int=None, chunk_size:int=None):
        if device is None or device < 0:
            device = sounddevice.default.device[0]
        print(f"device: {device}")

        sound_device = sounddevice.query_devices(device=device)
        print(f"sound_device: {sound_device}")

        if channels is None:
            channels = sound_device["max_input_channels"]

        if channels == 0:
            devices = sounddevice.query_devices()
            message = (
                "Selected input device does not have any input channels. \n"
                "Please set MicrophoneInterface(device=<YOUR DEVICE ID>). \n"
                f"Devices:\n{devices}"
            )
            raise IOError(message)

        if rate is None:
            rate = int(sound_device["default_samplerate"])

        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE

        print(f"Mic info: channels: {channels}, sample rate: {rate}, chunk size: {chunk_size}")

        super().__init__(device, channels, rate, chunk_size)


    def start_stream(self, device:int=-1):
        if device < 0:
            device = None
        super().start_stream()
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback,
            input_device_index=device,
        )
        self.stream.start_stream()
        

class HumeAudioStream(AudioStream):
    def __init__(self, device:int=-1, channels:int=1, rate:int=DEFAULT_SAMPLE_RATE, chunk_size:int=DEFAULT_CHUNK_SIZE, config_id:str=DEFAULT_HUME_CONFIG_ID):
        super().__init__(channels, rate, chunk_size)
        self.device = device
        self.config_id = config_id
        self.audio_queue = queue.Queue()
        self.thread = None
        self.buffer = b''  # Buffer to store leftover bytes
        self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer access

    def start_stream(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_hume_stream, daemon=True)
            self.thread.start()

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_stream(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()

    def _run_hume_stream(self):
        asyncio.run(hume_stream(device=self.device, config_id=self.config_id, audio_callback=self._audio_callback))

    def _audio_callback(self, audio_bytes:bytes):
        if not self.running:
            return
        
        # NOTE: audio_bytes is actually an audiofile, see play_audio()
        # and currently it is the entire file, not streaming!

        try:
            segment = AudioSegment.from_file(BytesIO(audio_bytes))
            
            # Calculate bytes per chunk
            bytes_per_chunk = self.chunk_size * segment.sample_width * segment.channels
            delay = bytes_per_chunk / segment.frame_rate

            samples = segment.get_array_of_samples()

            total_length = len(segment.raw_data)
            num_complete_chunks = total_length // bytes_per_chunk

            # Calculate end index for complete chunks
            end_index = num_complete_chunks * bytes_per_chunk

            # Extract complete chunks
            for i in range(num_complete_chunks):
                start = i * bytes_per_chunk
                end = start + bytes_per_chunk
                small_chunk = samples[start:end]
                self.audio_queue.put(small_chunk)
                #print(f"Enqueued chunk {i+1}/{num_complete_chunks}, Size: {len(small_chunk)} bytes")
                time.sleep(0.01) # delay
            
            # if remaining bytes, pad the end and send
            if end_index < total_length:
                small_chunk = samples[end_index:]
                small_chunk = np.pad(small_chunk, (0, bytes_per_chunk - len(small_chunk)), mode='constant')
                self.audio_queue.put(small_chunk)

        except Exception as e:
            print(f"Error in hume audio callback: {e}")
            self.running = False


# class PyHumeAudioStream(PyAudioStream):
#     def __init__(self, rate=DEFAULT_SAMPLE_RATE, chunk_size=DEFAULT_CHUNK_SIZE, device=-1, config_id=DEFAULT_HUME_CONFIG_ID):
#         super().__init__(rate, chunk_size)
#         self.device = device
#         self.config_id = config_id
#         self.audio_queue = queue.Queue()
#         self.thread = None
#         self.buffer = b''  # Buffer to store leftover bytes
#         self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer access

#     def start_stream(self):
#         if not self.running:
#             self.running = True
#             self.thread = threading.Thread(target=self._run_hume_stream, daemon=True)
#             self.thread.start()

#     def get_audio_chunk(self):
#         try:
#             return self.audio_queue.get_nowait()
#         except queue.Empty:
#             return None

#     def stop_stream(self):
#         if self.running:
#             self.running = False
#             if self.thread:
#                 self.thread.join()

#     def _run_hume_stream(self):
#         asyncio.run(hume_stream(device=self.device, config_id=self.config_id))

#     async def _play_callback(self, audio_chunk:bytes):
#         segment = AudioSegment.from_file(BytesIO(audio_chunk))
    
#         # self.stream = self.audio_interface.open(
#         #     format=pyaudio.paInt16,
#         #     channels=1,
#         #     rate=self.rate,
#         #     output=True,
#         #     frames_per_buffer=self.chunk_size,
#         #     stream_callback=self.callback,
#         # )
#         def play():
#             self.stream = self.audio_interface.open(format=self.audio_interface.get_format_from_width(segment.sample_width),
#                             channels=segment.channels,
#                             rate=segment.frame_rate,
#                             output=True, # FIXME: don't actually output to speakers
#                             stream_callback=self.callback,) 
#             self.stream.start_stream()

#         await asyncio.to_thread(play, segment)
        

#     def _audio_callback(self, audio_chunk:bytes):
#         if not self.running:
#             return
        
#         try:
#             segment = AudioSegment.from_file(BytesIO(audio_chunk))

#             # Define audio specifications
#             SAMPLE_RATE = segment.frame_rate    # e.g., 16000 Hz
#             SAMPLE_WIDTH = segment.sample_width # 16 bits = 2 bytes
#             CHANNELS = segment.channels

#             # Calculate bytes per chunk
#             bytes_per_chunk = self.chunk_size * SAMPLE_WIDTH * CHANNELS
#             delay = bytes_per_chunk / SAMPLE_RATE

#             # Acquire lock before modifying the buffer
#             with self.buffer_lock:
#                 # Prepend leftover bytes to the new audio_chunk
#                 combined_chunk = self.buffer + audio_chunk
#                 total_length = len(combined_chunk)
#                 num_complete_chunks = total_length // bytes_per_chunk

#                 # Calculate end index for complete chunks
#                 end_index = num_complete_chunks * bytes_per_chunk

#                 # Extract complete chunks
#                 for i in range(num_complete_chunks):
#                     start = i * bytes_per_chunk
#                     end = start + bytes_per_chunk
#                     small_chunk = combined_chunk[start:end]
#                     self.audio_queue.put(small_chunk)
#                     #time.sleep(delay)
#                     #print(f"Enqueued chunk {i+1}/{num_complete_chunks}, Size: {len(small_chunk)} bytes")

#                 # Store any remaining bytes in the buffer
#                 self.buffer = combined_chunk[end_index:]
#                 if self.buffer:
#                     pass
#                     #logger.debug(f"Buffering {len(self.buffer)} leftover bytes for next callback.")

#         except Exception as e:
#             print(f"Error in hume audio callback: {e}")
#             self.running = False