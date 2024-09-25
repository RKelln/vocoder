# Vocoder prototype

Experimental audio to multiple 0-255 output "vocoder".


## Install

Note: tested only on Unbuntu 22.04 wih python 3.11

Make a python environment:
```
python3.11 -m venv .venv
source .venv/bin/activate
```

Note: for Linux need:
```
$ sudo apt install python3.11-dev
```

Install packages:
```
$ pip install python-dotenv "hume[microphone]" pyaudio numpy pyaudio librosa rich scipy pydub webrtcvad sounddevice
```

## Run
To use with your microphone for testing:

```
$ python3.11 audio_processor.py --filter --vad
```

You can load custom chains using a json file:

```json
{
    "frequency": 0.5,
    "volume": 0.5,
    "pulse": {
        "weight": 1.0,
        "pulse_frequency": 0.8,
        "pulse_value": 100
    },
    "volume_random": 0.9
}
```

```
$ python audio_processor.py --filter --vad --config chain_config.json
```

You can also select one of the visualizers (see `visualizers.py`):

```
$ python audio_processor.py --visual list
```



## Install on Pi


```bash
sudo apt-get --yes update
sudo apt-get --yes install libasound2-dev libportaudio2 ffmpeg python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate

pip install python-dotenv "hume[microphone]" pyaudio numpy pyaudio librosa pydub scipy webrtcvad sounddevice RPi.GPIO gpiozero

```