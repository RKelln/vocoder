"""Async client for handling messages to and from an EVI connection."""

import asyncio
import base64
import datetime
import json
import logging
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional

import wave

from hume._voice.microphone.asyncio_utilities import Stream
from hume._voice.microphone.audio_utilities import play_audio
from hume._voice.microphone.microphone_sender import Sender
from hume._voice.voice_socket import VoiceSocket
from hume.error.hume_client_exception import HumeClientException

from hume._voice.microphone.microphone import Microphone
from hume._voice.microphone.microphone_interface import MicrophoneInterface
from hume._voice.microphone.microphone_sender import MicrophoneSender
from hume._voice.microphone.chat_client import ChatClient
from hume._voice.voice_socket import VoiceSocket

logger = logging.getLogger(__name__)

# audio callback type
AudioCallbackType = Optional[Callable[[bytes], None]]

VERBOSE_HUME = True

@dataclass
class CallbackMicrophoneInterface(MicrophoneInterface):
    """Custom Interface for connecting a device microphone to an EVI connection with callback support."""

    @classmethod
    async def start(
        cls,
        socket: VoiceSocket,
        device: Optional[int] = Microphone.DEFAULT_DEVICE,
        allow_user_interrupt: bool = MicrophoneInterface.DEFAULT_ALLOW_USER_INTERRUPT,
        audio_callback: AudioCallbackType = None,
        # overrides for autodetected values
        num_channels: Optional[int] = None, sample_rate:Optional[int] = None, chunk_size: Optional[int] = None
    ) -> None:
        """Start the microphone interface with a callback.

        Args:
            socket (VoiceSocket): EVI socket.
            device (Optional[int]): Device index for the microphone.
            allow_user_interrupt (bool): Whether to allow the user to interrupt EVI.
            callback (AudioCallbackType): Function to call with each audio chunk returned.
        """
        if device < -1:
            device = None
        
        with Microphone.context(device=device, num_channels=num_channels, sample_rate=sample_rate, chunk_size=chunk_size) as microphone:
            sender = MicrophoneSender.new(microphone=microphone, allow_interrupt=allow_user_interrupt)
            #sender = MicrophoneFileSaver.new(microphone=microphone, file_path="mic_file.wav", allow_interrupt=allow_user_interrupt)
            chat_client = CallbackChatClient.new(sender=sender)
            print("Configuring socket with microphone settings...")
            await socket.update_session_settings(
                sample_rate=microphone.sample_rate,
                num_channels=microphone.num_channels,
            )
            print("Microphone connected. Say something!")
            await chat_client.run(socket=socket, audio_callback=audio_callback)



@dataclass
class CallbackChatClient(ChatClient):

    audio_callback: AudioCallbackType = None
    play_callback: AudioCallbackType = None

    async def _recv(self, *, socket: VoiceSocket, audio_callback: AudioCallbackType = None, verbose:bool=False) -> None:
        if audio_callback is None:
            audio_callback = self.audio_callback
        
        async for socket_message in socket:
            message = json.loads(socket_message)
            if message["type"] in ["user_message", "assistant_message"]:
                role = self._map_role(message["message"]["role"])
                message_text = message["message"]["content"]
                text = f"{role}: {message_text}"
            elif message["type"] == "audio_output":
                message_str: str = message["data"]
                message_bytes = base64.b64decode(message_str.encode("utf-8"))
                if audio_callback is not None:
                    audio_callback(message_bytes)
                else:
                    await self.byte_strs.put(message_bytes)
                continue
            elif message["type"] == "error":
                error_message: str = message["message"]
                error_code: str = message["code"]
                raise HumeClientException(f"Error ({error_code}): {error_message}")
            elif message["type"] == "tool_call":
                print(
                    "Warning: EVI is trying to make a tool call. "
                    "Either remove tool calling from your config or "
                    "use the VoiceSocket directly without a MicrophoneInterface."
                )
                tool_call_id = message["tool_call_id"]
                if message["response_required"]:
                    content = "Let's start over"
                    await self.sender.send_tool_response(socket=socket, tool_call_id=tool_call_id, content=content)
                continue
            elif message["type"] == "chat_metadata":
                message_type = message["type"].upper()
                chat_id = message["chat_id"]
                chat_group_id = message["chat_group_id"]
                text = f"<{message_type}> Chat ID: {chat_id}, Chat Group ID: {chat_group_id}"
            else:
                message_type = message["type"].upper()
                text = f"<{message_type}>"

            if verbose:
                self._print_prompt(text)

    async def _play(self) -> None:
        async for byte_str in self.byte_strs:
            await self.sender.on_audio_begin()
            if self.play_callback is not None:
                await self.play_callback(byte_str)
            else:
                await play_audio(byte_str)
            await self.sender.on_audio_end()


    async def run(self, *, socket: VoiceSocket, 
                  audio_callback: AudioCallbackType = None, 
                  play_callback: AudioCallbackType = None,
                  verbose:bool = False) -> None:
        """Run the chat client.

        Args:
            socket (VoiceSocket): EVI socket.
        """
        if play_callback is not None:
            self.play_callback = play_callback
        
        recv = self._recv(socket=socket, audio_callback=audio_callback, verbose=VERBOSE_HUME)
        send = self.sender.send(socket=socket)

        # if audio callback is provided, then we don't play the audio stream
        if audio_callback is not None:
            await asyncio.gather(recv, send)
        else:
            await asyncio.gather(recv, self._play(), send)


@dataclass
class MicrophoneFileSaver(Sender):
    """Saves microphone audio data to a file instead of sending over a socket."""

    microphone: Microphone
    file_path: str
    send_audio: bool
    allow_interrupt: bool

    # We need to keep track of the file object
    file: Optional[object] = None  # Initialized as None

    @classmethod
    def new(cls, *, microphone: Microphone, file_path: str, allow_interrupt: bool) -> "MicrophoneFileSaver":
        """Create a new MicrophoneFileSaver.

        Args:
            microphone (Microphone): Microphone instance.
            file_path (str): Path to the output file where audio data will be saved.
            allow_interrupt (bool): Whether to allow interrupting the audio stream.
        """
        print(f"Creating new MicrophoneFileSaver with file path: {file_path}")
        return cls(microphone=microphone, file_path=file_path, send_audio=True, allow_interrupt=allow_interrupt)

    async def on_audio_begin(self) -> None:
        """Handle the start of an audio stream."""
        self.send_audio = self.allow_interrupt
        # Open the file for writing in binary mode
        self.file = open(self.file_path, 'wb')
        logger.debug(f"Opened file {self.file_path} for writing audio data.")

    async def on_audio_end(self) -> None:
        """Handle the end of an audio stream."""
        self.send_audio = True
        if self.file:
            # Close the file
            self.file.close()
            logger.debug(f"Closed file {self.file_path} after writing audio data.")
            self.file = None

    async def send(self, *, socket: VoiceSocket) -> None:
        """Save audio data to a file instead of sending over an EVI socket.

        Args:
            socket (VoiceSocket): EVI socket (not used in this implementation).
        """

        with wave.open(self.file_path, 'wb') as wav_file:
            wav_file.setnchannels(self.microphone.num_channels)
            wav_file.setsampwidth(self.microphone.sample_width)
            wav_file.setframerate(self.microphone.sample_rate)
            
            async for byte_str in self.microphone:
                if self.send_audio:
                    # Write the audio bytes to the file
                    #byte_str = bytes(byte_str)
                    nonzero = any(byte_str)  # Ensure the byte string is not empty
                    print(f"Received audio chunk: {len(byte_str)} bytes, nonzero_bytes: {nonzero}")
                    wav_file.writeframes(byte_str)
                    # Optionally, flush to ensure data is written promptly
                    await asyncio.sleep(0)  # Yield control to ensure responsiveness

    async def send_tool_response(self, *, socket: VoiceSocket, tool_call_id: str, content: str) -> None:
        """Handle tool responses (not applicable for file saving).

        Args:
            socket (VoiceSocket): EVI socket (not used in this implementation).
            tool_call_id (str): Tool call ID.
            content (str): Tool response content.
        """
        # Since we're not using the socket, we can log the response
        logger.debug(f"Tool response: {{'tool_call_id': {tool_call_id}, 'content': {content}}}")
