import argparse
import time
import sys
import json
import os
import importlib

from audio_stream import DEFAULT_SAMPLE_RATE, AudioStream, MicAudioStream, PyAudioFileAudioStream, HumeAudioStream, VoiceActivityDetector
from translators import get_translators, AlgorithmChain
from visualizers import get_visualizers, Visualizer, BarVisualizer, BubbleVisualizer

print("Loading...")
for translator in get_translators().values():
    print(translator)
    importlib.import_module("translators", str(translator))

for visualizer in get_visualizers().values():
    print(visualizer)
    importlib.import_module("visualizers", str(visualizer))

"""
Audio Processor Module
This module provides classes and functions for processing audio streams, applying filters, and visualizing the output.
Classes:
    ParameterInput:
        Manages parameters for audio processing, such as sentiment.
    AudioProcessor:
        Processes audio streams, applies algorithms, and visualizes the output.
Functions:
    high_pass_filter(audio_chunk, cutoff=100, fs=44100):
        Applies a high-pass filter to the given audio chunk.
Usage:
    Run the module as a script to start the audio processor with optional command-line arguments:
        --vad: Use Voice Activity Detection
        --filter: Apply high-pass filter
Example:
    python audio_processor.py --vad --filter
"""

class ParameterInput:
    def __init__(self):
        self.parameters = {'sentiment': 0.5}  # Default sentiment value (0.0 sad, 1.0 happy)

    def get_parameters(self):
        # For now, return static parameters
        return self.parameters

    def set_sentiment(self, value):
        self.parameters['sentiment'] = value


def high_pass_filter(audio_chunk, cutoff:float=100, fs:float=44100):
    '''
    Apply a high-pass filter to the given audio chunk.
    Parameters:
        audio_chunk (np.ndarray): Audio data as a 1D NumPy array.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
    Returns:
        np.ndarray: Filtered audio data.
    '''

    from scipy.signal import butter, sosfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(N=5, Wn=normal_cutoff, btype='highpass', output='sos')
    filtered_audio = sosfilt(sos, audio_chunk)
    return filtered_audio


class AudioProcessor:
    num_outputs:int
    audio_stream:AudioStream
    sampling_rate:int
    output_visualizer:Visualizer

    def __init__(self, audio_stream, 
                 sampling_rate=DEFAULT_SAMPLE_RATE, 
                 num_outputs=5, use_vad=False, 
                 high_pass_filter=False, 
                 chain=None, 
                 visualizer=BarVisualizer,
                 fps=30.0):
        self.num_outputs = num_outputs
        self.audio_stream = audio_stream
        self.sampling_rate = sampling_rate
        self.parameter_input = ParameterInput()
        self.output_visualizer = visualizer(num_outputs)
        self.algorithm_chain = AlgorithmChain()
        self.setup_algorithms(chain)

        self.use_vad = use_vad
        self.high_pass_filter = high_pass_filter

        if self.use_vad:
            self.voice_activity_detector = VoiceActivityDetector()

        self.delay = 1.0 / fps
        

    def setup_algorithms(self, chain_dict:dict):
        parameters = self.parameter_input.get_parameters()
        
        if chain_dict is None:
            chain_dict = {'frequency': 1.0}

        for translator, data in chain_dict.items():
            if translator in get_translators():
                translator_class = get_translators()[translator]
                translator_params = parameters.copy()
                if isinstance(data, float):
                    weight = data
                elif isinstance(data, dict):
                    weight = data.get('weight', 1.0)
                    translator_params.update(data)
                    print(f"Parameters for {translator}: {translator_params}")
                else:
                    raise ValueError("Invalid data format for translator.")
                algorithm = translator_class(self.num_outputs, parameters=translator_params)
                self.algorithm_chain.add_algorithm(algorithm, weight=weight)
                print(f"Added {translator} {algorithm} with weight {weight}.")
            else:
                print("Invalid translator: ", translator)

        self.algorithm_chain.update_weights() # normalize weights to sum to 1.0


    def update_algorithm_weights(self):
        return # skip for now
        # Example: Adjust weights based on sentiment parameter
        parameters = self.parameter_input.get_parameters()
        sentiment = parameters['sentiment']
        # Sentiment ranges from 0.0 (sad) to 1.0 (happy)
        # Let's say higher sentiment increases the weight of the frequency algorithm
        freq_weight = sentiment
        volume_weight = 1.0 - sentiment
        self.algorithm_chain.update_weights([freq_weight, volume_weight])


    def run(self):
        chunk_duration = self.audio_stream.get_chunk_duration()

        try:
            with self.audio_stream, self.output_visualizer:
                next_process_time = time.monotonic()
                while self.audio_stream.running:
                    t = time.monotonic()
                    audio_chunk = self.audio_stream.get_audio_chunk()
                    if audio_chunk is not None:
                        #print("got audio chunk", len(audio_chunk))
                        if self.high_pass_filter:
                            # Apply high-pass filter
                            filtered_chunk = high_pass_filter(audio_chunk, fs=self.sampling_rate)
                        else:
                            filtered_chunk = audio_chunk
                        
                        self.update_algorithm_weights()

                        if self.use_vad:
                            # Update voice activity detector (if using dynamic threshold)
                            #self.voice_activity_detector.update_noise_floor(filtered_chunk)

                            outputs = self.algorithm_chain.process(filtered_chunk, 
                                                                    is_speech=self.voice_activity_detector.is_speech(filtered_chunk))

                        else:  # No VAD
                            outputs = self.algorithm_chain.process(filtered_chunk, is_speech=True)

                        # Display outputs
                        self.output_visualizer.display(outputs)

                        # sleep if needed
                        next_process_time += chunk_duration
                        now = time.monotonic()
                        delay = next_process_time - now
                        if delay < 0:
                            next_process_time = now
                    else:
                        delay = max(0.01, self.delay - (time.monotonic() - t))
                    
                    if delay > 0:
                        time.sleep(delay)  # Sleep briefly to reduce CPU usage
        except KeyboardInterrupt:
            print("Stream stopped by user.")
        except Exception as e:
            print(f"An error occurred: {e}")


def load_chain_settings(config_path):
    with open(config_path, 'r') as file:
        chain = json.load(file)
    return chain


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Audio Processor")
    parser.add_argument("-n", "--num_outputs", "--num-outputs", type=int, default=5, help="Number of output values")
    parser.add_argument("--visualizer", "--visual", type=str, choices=list(get_visualizers().keys()), default='bubble', help="Visualizer type")
    parser.add_argument("--vad", action="store_true", help="Use Voice Activity Detection")
    parser.add_argument("--filter", action="store_true", help="Apply high-pass filter")
    parser.add_argument("--chain", type=str, help="Path to the chain configuration file.")
    parser.add_argument("--audio", type=str, help="Path to an audio file (.wav, .mp3) for testing.")
    parser.add_argument("-r", "--rate", "--sampling_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sampling rate for audio stream.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice
        print(sounddevice.query_devices())
        exit(0)

    chain = {'volume_random': 1.0}
    chain = {'volume_random': 
                {
                    "weight": 1.0,
                    "dynamic_range": 40,  # update the dynamic range 40 times
                    "min_input": 1000, # because dynamic, start large
                    "max_input": 100,  # because dynamic, start small
                }
            }   

    if args.chain:
        if not args.chain.endswith('.json'):
            print("Chain configuration file must be a JSON file.")
            sys.exit(1)
        if not os.path.exists(args.chain):
            print("Chain configuration file not found.", args.chain)
            sys.exit(1)
        chain = load_chain_settings(args.chain)
    
    print("Chain settings:")
    print(chain)

    # init visualizers
    visualizers = get_visualizers()
    visualizer = visualizers.get(args.visualizer.lower(), BubbleVisualizer)

    # init audio
    if args.audio:
        if args.audio.startswith("hume"):
            # split on : for config id
            if ":" not in args.audio:
                print("Hume config id not found, using default. You can set this using --audio hume:<config_id>")
                audio_stream = HumeAudioStream(device=args.device)
            else:
                config_id = args.audio.split(":")[1]
                audio_stream = HumeAudioStream(device=args.device, config_id=config_id)
        elif args.audio == "mic":
            audio_stream = MicAudioStream(device=args.device)
        else:
            if not os.path.exists(args.audio):
                print("Audio file not found.", args.audio)
                sys.exit
            
            audio_stream = PyAudioFileAudioStream(args.audio, rate=args.rate, loop=True)
    else:
        audio_stream = MicAudioStream(device=args.device)

    processor = AudioProcessor(audio_stream, sampling_rate=args.rate, num_outputs=args.num_outputs, use_vad=args.vad, high_pass_filter=args.filter,
                               chain=chain, visualizer=visualizer, fps=30)
    processor.run()