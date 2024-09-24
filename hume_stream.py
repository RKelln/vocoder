import os
import asyncio
import argparse

from hume import HumeVoiceClient, MicrophoneInterface
from hume_callback_client import CallbackMicrophoneInterface, AudioCallbackType
from dotenv import load_dotenv

# See: https://github.com/HumeAI/hume-api-examples/tree/main/evi-python-example

DEFAULT_HUME_CONFIG_ID = "d2868b61-e3ad-4d71-b3d5-5ee091672d04"


async def hume_stream(device:int = -1, config_id:str=DEFAULT_HUME_CONFIG_ID, audio_callback:AudioCallbackType = None ) -> None:
    load_dotenv()

    # Retrieve the Hume API key from the environment variables
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    # Connect and authenticate with Hume
    client = HumeVoiceClient(HUME_API_KEY)

    # Start streaming EVI over your device's microphone and speakers
    async with client.connect(config_id=config_id) as socket:
        if device == -1:
            await CallbackMicrophoneInterface.start(socket, allow_user_interrupt=True, 
                                                    audio_callback=audio_callback)   
        else:   
            await CallbackMicrophoneInterface.start(socket, device=device, allow_user_interrupt=True, 
                                                    audio_callback=audio_callback)


async def mic_test(device:int = -1):
    from hume._voice.microphone.microphone import Microphone

    if device < 0:
        print("Using default microphone device")
        device = None

    with Microphone.context(device=device) as microphone:
        print(f"Microphone sample rate: {microphone.sample_rate}")
        print(f"Microphone channels: {microphone.num_channels}")
        async for audio_chunk in microphone:
            print(f"Received audio chunk: {len(audio_chunk)} bytes")
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("HUME_API_KEY"):
        raise ValueError("HUME_API_KEY not found in environment variables")
    if not os.getenv("HUME_SECRET_KEY"):
        raise ValueError("HUME_SECRET_KEY not found in environment variables")
    
    # parse args for --list-deices and display sound devices
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--test-mic", "--test_mic", action="store_true")
    parser.add_argument("--config-id", type=str, default=DEFAULT_HUME_CONFIG_ID)
    parser.add_argument("--audio-callback", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice
        print(sounddevice.query_devices())
        exit(0)

    if args.test_mic:
        asyncio.run(mic_test(args.device))
        exit(0)

    audio_callback = None
    if args.audio_callback:
        def audio_callback(audio_chunk):
            print(f"Received audio chunk: {len(audio_chunk)} bytes")
        audio_callback = audio_callback

    asyncio.run(hume_stream(args.device, config_id=args.config_id, audio_callback=audio_callback))