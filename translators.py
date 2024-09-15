import numpy as np

from audio_stream import hz_to_mel, mel_to_hz, DEFAULT_SAMPLE_RATE, DEFAULT_CHUNK_SIZE

def get_translators():
    return {
        "silence": SilenceAlgorithm,
        "frequency": FrequencyBinAlgorithm,
        "volume": VolumeOverTimeAlgorithm,
        "volume_time": VolumeOverTimeAlgorithm,
        "pulse": PulseOverTimeGenerator,
        "volume_random": VolumeRandomOutputAlgorithm,
    } 

class Algorithm:
    max_output_value = 255
    default_min_input:float = 0.0
    default_max_input:float = 1.0

    def __init__(self, num_outputs, parameters=None):
        self.num_outputs = num_outputs
        assert self.num_outputs > 0, "Number of outputs must be greater than zero."
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        self.min_input = self.parameters.get('min_input', self.default_min_input)
        self.max_input = self.parameters.get('max_input', self.default_max_input)
        self.parameters['min_input'] = self.min_input
        self.parameters['max_input'] = self.max_input

        self.input_range = self.max_input - self.min_input
        assert self.input_range > 0, "Minimum input must be less than maximum input."

    def process(self, audio_chunk, is_speech=True):
        raise NotImplementedError("Algorithm process method must be implemented.")

    def scale_outputs(self, outputs):
        outputs = np.nan_to_num(outputs, nan=0.0)  # Replace NaN with zero
        #scaled = [int(min(self.input_range, max(0, value - self.min_input)) / self.input_range * self.max_output_value) for value in outputs]
        
        # Clip outputs to the specified range
        outputs = np.clip(outputs, self.min_input, self.max_input)

        # Interpolate the clipped values to the range [0, self.max_output_value]
        scaled = np.interp(outputs, (self.min_input, self.max_input), (0, self.max_output_value))

        for i, value in enumerate(scaled):
            if value >= self.max_output_value:
                print(f"Clip warning: {i}: {outputs[i]} vs max input: {self.max_input}.")
        
        return list(scaled)
    
    def silence(self):
        return [0] * self.num_outputs

    def __str__(self) -> str:
        return self.__class__.__name__ + f"{self.parameters}"
    

class SilenceAlgorithm(Algorithm):
    def process(self, audio_chunk):
        return self.silence()


class LogFrequencyBinAlgorithm(Algorithm):
    # focus on human speech fundamental frequencies (100 Hz to 500 Hz)
    default_min_freq:float = 100.0
    default_max_freq:float = 8000.0
    default_min_input:float = 200.0
    default_max_input:float = 30000.0

    def __init__(self, num_outputs, parameters=None):
        super().__init__(num_outputs, parameters)
        # setup frequency bins
        total_bins = self.num_outputs
        min_freq = self.parameters.get('min_freq', self.default_min_freq)
        max_freq = self.parameters.get('max_freq', self.default_max_freq)
        self.parameters['min_freq'] = min_freq
        self.parameters['max_freq'] = max_freq

        self.bin_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num=total_bins+1, endpoint=True, base=10.0, dtype=int)


    def process(self, audio_chunk, is_speech=True):
        # Apply FFT to get frequency components
        #fft_result = np.fft.rfft(audio_chunk)
        
        window = np.hamming(len(audio_chunk))
        windowed_chunk = audio_chunk * window
        fft_result = np.fft.rfft(windowed_chunk)
        magnitude = np.abs(fft_result)

        # Check if magnitude array is empty
        if len(magnitude) == 0:
            outputs = self.silence()
        else:
            if len(magnitude) < self.bin_edges[-1]:
                # Pad with zeros if too short
                magnitude = np.pad(magnitude, (0, self.bin_edges[-1] - len(magnitude)), 'constant')

            # Split into frequency bins
            outputs = []
            for i in range(self.num_outputs):
                start = self.bin_edges[i]
                end = self.bin_edges[i+1]
                bin_slice = magnitude[start:end]
                if len(bin_slice) == 0:
                    bin_value = 0
                else:
                    bin_value = np.nanmean(bin_slice)
                outputs.append(bin_value)
        #print(max(outputs))
        # Normalize and scale outputs to 0-255
        outputs = self.scale_outputs(outputs)
        return outputs


class MelFrequencyBinAlgorithm(Algorithm):
    default_min_freq: float = 100.0
    default_max_freq: float = 8000.0
    default_min_input: float = 0.5   # Set based on expected minimum output after processing
    default_max_input: float = 10.0  # Set based on expected maximum output after processing
    noise_suppression_samples:int = 10 

    def __init__(self, num_outputs, parameters=None):
        super().__init__(num_outputs, parameters)
        self.sample_rate = parameters.get('sample_rate', DEFAULT_SAMPLE_RATE)
        self.parameters['sample_rate'] = self.sample_rate
        self.chunk_size = parameters.get('chunk_size', DEFAULT_CHUNK_SIZE)  # Fixed chunk size
        self.parameters['chunk_size'] = self.chunk_size
        self.N = self.chunk_size  # FFT size

        min_freq = self.parameters.get('min_freq', self.default_min_freq)
        max_freq = self.parameters.get('max_freq', self.default_max_freq)
        self.parameters['min_freq'] = min_freq
        self.parameters['max_freq'] = max_freq

        # Calculate Mel-frequency bin edges
        min_mel = hz_to_mel(min_freq)
        max_mel = hz_to_mel(max_freq)
        mel_edges = np.linspace(min_mel, max_mel, num=self.num_outputs + 1)
        freq_edges = mel_to_hz(mel_edges)
        self.freq_edges = freq_edges  # Store for debugging
        self.bin_widths_hz = self.freq_edges[1:] - self.freq_edges[:-1]

        # Convert frequencies to FFT bin indices
        self.bin_indices = np.floor(freq_edges * self.N / self.sample_rate).astype(int)
        self.bin_indices = np.clip(self.bin_indices, 0, self.N // 2)  # Valid indices for rfft
        print("Mel edges:", mel_edges)
        print("Freq edges:", freq_edges)
        print("Bin indices:", self.bin_indices)
        #self.bin_indices = [0, 4, 8, 16, 32, 128]

        # noise floor / removal
        self.noise_spectrum = np.zeros(self.N // 2 + 1)
        self.noise_spectrum_samples = []


    def process(self, audio_chunk, is_speech=True):
        # Ensure the chunk size matches the expected size
        if len(audio_chunk) != self.N:
            # Optionally handle mismatched chunk sizes
            audio_chunk = audio_chunk[:self.N] if len(audio_chunk) > self.N else np.pad(audio_chunk, (0, self.N - len(audio_chunk)), 'constant')

        # Apply window function to reduce spectral leakage
        window = np.hamming(self.N)
        windowed_chunk = audio_chunk * window
        fft_result = np.fft.rfft(windowed_chunk)
        magnitude = np.abs(fft_result)

        # Check if magnitude array is empty
        if len(magnitude) == 0:
            return self.silence()

        if not is_speech:
            self.update_noise_spectrum(magnitude)
            return self.silence()

        # Apply noise suppression
        magnitude = magnitude - self.noise_spectrum
        magnitude = np.maximum(magnitude, 0)

        outputs = []
        for i in range(self.num_outputs):
            start = self.bin_indices[i]
            end = self.bin_indices[i + 1]
            bin_slice = magnitude[start:end]
            #print("Bin slice:", i, bin_slice)
            if len(bin_slice) == 0:
                bin_value = 0
            else:
                bin_width_hz = self.bin_widths_hz[i]
                if bin_width_hz == 0:
                    bin_value = 0
                else:
                    bin_value = np.sum(bin_slice) / bin_width_hz
            outputs.append(bin_value)

        # Apply logarithmic scaling to compress dynamic range
        outputs = np.log1p(outputs)
        #print("Outputs:", outputs)

        # Scale outputs using fixed min_input and max_input
        outputs = self.scale_outputs(outputs)
        return outputs

    def update_noise_spectrum(self, magnitude):
        self.noise_spectrum_samples.append(magnitude)
        if len(self.noise_spectrum_samples) > self.noise_suppression_samples:
            self.noise_spectrum_samples.pop(0)
        self.noise_spectrum = np.mean(self.noise_spectrum_samples, axis=0)

FrequencyBinAlgorithm = MelFrequencyBinAlgorithm



class OverTimeAlgorithm(Algorithm):
    default_duration:float = 0.8
    default_sample_rate:float = 44100

    def __init__(self, num_outputs, parameters):
        super().__init__(num_outputs, parameters)
        self.history =  self.silence()
        self.direction = self.parameters.get('direction', 1)
        self.parameters['direction'] = self.direction
        self.duration = self.parameters.get('duration', self.default_duration)
        self.parameters['duration'] = self.duration
        self.sample_rate = self.parameters.get('sample_rate', self.default_sample_rate)
        self.parameters['sample_rate'] = self.sample_rate
        self.time_per_bin = self.duration / self.num_outputs
        self.cumulative_time = 0

    def process(self, audio_chunk, is_speech=True):
        if len(audio_chunk) > 0:
            # chunk size to time
            time_inc = len(audio_chunk) / self.sample_rate
            self.cumulative_time += time_inc

            value = self._process(audio_chunk, time_inc)
                
            # update time bins if enough time has passed
            # otherwie take max of current and new value
            if self.cumulative_time > self.time_per_bin:
                self.cumulative_time = self.cumulative_time - self.time_per_bin
                if self.direction == -1:
                    self.history.insert(0, value)
                    self.history.pop(-1)
                else:
                    self.history.append(value)
                    self.history.pop(0)
            else:
                self.history[-1] = max(self.history[-1], value)

        outputs = self.scale_outputs(self.history)
        return outputs

    def _process(self, audio_chunk, time_inc):
        raise NotImplementedError("OverTimeAlgorithm _process method must be implemented.")


class VolumeOverTimeAlgorithm(OverTimeAlgorithm):
    default_min_input:float = 0.0
    default_max_input:float = 3500.0

    def _process(self, audio_chunk, time_inc):
        # Calculate RMS (Root Mean Square) volume
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        if np.isnan(rms):
            rms = 0
        return rms


class PulseOverTimeGenerator(OverTimeAlgorithm):
    # generates a repeating pulse over time
    default_pulse_frequency:float = 1.0
    default_pulse_value:float = 255.0
    default_max_input:float = 255.0

    def __init__(self, num_outputs, parameters):
        super().__init__(num_outputs, parameters)
        self.pulse_frequency = self.parameters.get('pulse_frequency', self.default_pulse_frequency)
        self.parameters['pulse_frequency'] = self.pulse_frequency
        self.pulse_value = self.parameters.get('pulse_value', self.default_pulse_value)
        self.parameters['pulse_value'] = self.pulse_value

        self.pulse_timer = 0

    def _process(self, audio_chunk, time_inc):
        self.pulse_timer += time_inc
        if self.pulse_timer >= self.pulse_frequency:
            # send pulse
            self.pulse_timer = self.pulse_timer - self.pulse_frequency
            return self.pulse_value
        return 0


class VolumeRandomOutputAlgorithm(Algorithm):
    default_min_input:float = 100.0
    default_max_input:float = 1500.0
    default_silence_threshold:float = 130.0

    def __init__(self, num_outputs, parameters):
        super().__init__(num_outputs, parameters)
        self.silence_threshold = self.parameters.get('silence_threshold', self.default_silence_threshold)
        self.parameters['silence_threshold'] = self.silence_threshold
        self.current_output = 0
        self.waiting = True

    def process(self, audio_chunk, is_speech=True):
        # if the volume is low enough then pick a new output

        # Calculate RMS (Root Mean Square) volume
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        if np.isnan(rms):
            rms = 0
        if rms >= self.silence_threshold:
            self.waiting = False
        else:
            rms = 0

        if not self.waiting and (rms < self.silence_threshold or is_speech == False):
            self.current_output = np.random.randint(0, self.num_outputs)
            rms = 0
            self.waiting = True
        outputs = [rms if i == self.current_output else 0 for i in range(self.num_outputs)]
        return self.scale_outputs(outputs)


class AlgorithmChain:
    def __init__(self):
        self.algorithms = []
        self.weights = []

    def add_algorithm(self, algorithm, weight=1.0):
        self.algorithms.append(algorithm)
        self.weights.append(weight)

    def process(self, audio_chunk, is_speech=True):
        combined_outputs = np.zeros(self.algorithms[0].num_outputs)
        for algorithm, weight in zip(self.algorithms, self.weights):
            if weight == 0:
                continue
            outputs = algorithm.process(audio_chunk, is_speech=is_speech)
            weighted_outputs = np.array(outputs) * weight
            combined_outputs += weighted_outputs

        # Replace NaN in combined_outputs with zeros
        combined_outputs = np.nan_to_num(combined_outputs, nan=0.0)

        # Normalize combined outputs to 0-255
        #max_value = max(combined_outputs) or 1
        #outputs = [int((value / max_value) * 255) for value in combined_outputs]
        return [int(value) for value in combined_outputs]

    def update_weights(self, weights=None):
        if weights is not None:
            assert len(weights) == len(self.weights), "Number of weights must match number of algorithms."
            self.weights = weights
        # ensure weights um to 1.0
        total_weight = sum(self.weights)
        self.weights = [weight / total_weight for weight in self.weights]
