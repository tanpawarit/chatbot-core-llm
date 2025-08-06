from datetime import datetime, timezone
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from .message_model import Message


class Conversation(BaseModel):
    user_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        return self.messages[-limit:] if self.messages else []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }