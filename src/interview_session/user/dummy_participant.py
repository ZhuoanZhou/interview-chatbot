import os
import threading
import time
import asyncio

from typing import TYPE_CHECKING, Optional, List, Dict, Any

from src.interview_session.user.user import User
from src.interview_session.session_models import Message, MessageType
from src.utils.speech.speech_to_text import create_stt_engine


if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession

class UserDummyParticipant(User):
    def __init__(self, user_id: str, interview_session: 'InterviewSession'):
        super().__init__(user_id=user_id, interview_session=interview_session)
        self._message_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    async def on_message(self, message: Message):
        # Wait until add_user_message is called
        if message.role == "Interviewer":
            with self._lock:
                self._message_buffer.append({
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "type": message.type,
                    "timestamp": message.timestamp.isoformat()
                })

    def get_and_clear_messages(self):
        with self._lock:
            if not self._message_buffer:
                return []
            
            # Return a copy and clear the buffer
            messages = self._message_buffer[:]
            self._message_buffer.clear()
            return messages
        # while not hasattr(self, "_pending_user_message") or self._pending_user_message is None:
        #     await asyncio.sleep(0.1)
        
        # # Once message is ready, push to chat history
        # self.interview_session.add_message_to_chat_history(
        #     role="User",
        #     content=self._pending_user_message
        # )
        
        # # Clear pending message
        # self._pending_user_message = None

    def add_user_message(self, text: str):
        self.interview_session.add_message_to_chat_history(
            role="User",
            content=text
        )

    def get_interviewer_message(self):
        pass
