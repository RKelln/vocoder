import argparse
import threading
import time
import sys
import json
import os

import pyaudio
import numpy as np
import scipy

from audio_stream import DEFAULT_SAMPLE_RATE, MicAudioStream, PyAudioFileAudioStream, VoiceActivityDetector
from translators import get_translators, AlgorithmChain, VolumeOverTimeAlgorithm
from visualizer import get_visualizers, BarVisualizer, BubbleVisualizer

for translator in get_translators().values():
    print(translator)
    import translators

for visualizer in get_visualizers().values():
    print(visualizer)
    import visualizer

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
    def __init__(self, audio_stream, sampling_rate=DEFAULT_SAMPLE_RATE, num_outputs=5, use_vad=False, high_pass_filter=False, chain=None, visualizer=BarVisualizer):
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
        try:
            with self.audio_stream, self.output_visualizer:
                while True:
                    audio_chunk = self.audio_stream.get_audio_chunk()
                    if audio_chunk is not None:
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
                    else:
                        time.sleep(0.01)  # Sleep briefly to reduce CPU usage
        except KeyboardInterrupt:
            print("Stopping...")


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
    args = parser.parse_args()

    chain = {'volume_random': 1.0}
    chain = {'volume_random': 
                {
                    "wieght": 1.0,
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
        audio_stream = PyAudioFileAudioStream(args.audio, rate=args.rate, loop=True)
    else:
        audio_stream = MicAudioStream(rate=args.rate)

    processor = AudioProcessor(audio_stream, sampling_rate=args.rate, num_outputs=args.num_outputs, use_vad=args.vad, high_pass_filter=args.filter,
                               chain=chain, visualizer=visualizer)
    processor.run()