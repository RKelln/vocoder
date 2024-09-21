"""Async client for handling messages to and from an EVI connection."""

import asyncio
import base64
import datetime
import json
import logging
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional

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

VERBOSE_HUME = False

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
    ) -> None:
        """Start the microphone interface with a callback.

        Args:
            socket (VoiceSocket): EVI socket.
            device (Optional[int]): Device index for the microphone.
            allow_user_interrupt (bool): Whether to allow the user to interrupt EVI.
            callback (AudioCallbackType): Function to call with each audio chunk returned.
        """
        with Microphone.context(device=device) as microphone:
            sender = MicrophoneSender.new(microphone=microphone, allow_interrupt=allow_user_interrupt)
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
        self.play_callback = audio_callback

        recv = self._recv(socket=socket, audio_callback=audio_callback, verbose=VERBOSE_HUME)
        send = self.sender.send(socket=socket)

        # if audio callback is provided, then we don't play the audio stream
        if audio_callback is not None:
            await asyncio.gather(recv, send)
        else:
            await asyncio.gather(recv, self._play(), send)
