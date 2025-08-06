from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field

from src.models.base import MessageRole, MediaType


class MediaContent(BaseModel):
    """Multimodal content that can contain text and/or media files"""
    text: Optional[str] = None
    image_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    
    @property
    def media_type(self) -> MediaType:
        """Determine primary media type"""
        if self.image_path:
            return MediaType.IMAGE
        elif self.audio_path:
            return MediaType.AUDIO
        elif self.video_path:
            return MediaType.VIDEO
        else:
            return MediaType.TEXT
    
    @property
    def has_media(self) -> bool:
        """Check if content has any media files"""
        return bool(self.image_path or self.audio_path or self.video_path)
    
    @property
    def media_path(self) -> Optional[Path]:
        """Get the primary media file path"""
        return self.image_path or self.audio_path or self.video_path


class Message(BaseModel):
    role: MessageRole
    content: MediaContent
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def text_message(cls, role: MessageRole, text: str) -> 'Message':
        """Create a text-only message"""
        return cls(
            role=role,
            content=MediaContent(text=text)
        )
    
    @classmethod  
    def media_message(cls, role: MessageRole, text: Optional[str] = None, 
                     image_path: Optional[Path] = None,
                     audio_path: Optional[Path] = None,
                     video_path: Optional[Path] = None) -> 'Message':
        """Create a multimedia message"""
        return cls(
            role=role,
            content=MediaContent(
                text=text,
                image_path=image_path,
                audio_path=audio_path,
                video_path=video_path
            )
        )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v) if v else None
        }