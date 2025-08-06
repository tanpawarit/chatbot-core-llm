"""Media types and format definitions"""

from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class SupportedFormat(str, Enum):
    # Image formats
    JPEG = "image/jpeg"
    PNG = "image/png"  
    GIF = "image/gif"
    WEBP = "image/webp"
    
    # Audio formats
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    OGG = "audio/ogg"
    M4A = "audio/mp4"
    
    # Video formats
    MP4 = "video/mp4"
    WEBM = "video/webm"
    AVI = "video/avi"


class MediaInfo(BaseModel):
    """Media file information"""
    file_path: Path
    media_type: str  # image/audio/video
    format: SupportedFormat
    file_size: int  # in bytes
    is_valid: bool
    error_message: Optional[str] = None
    
    @property
    def size_mb(self) -> float:
        """File size in MB"""
        return self.file_size / (1024 * 1024)


# File extension to format mapping
EXTENSION_MAP = {
    # Images
    '.jpg': SupportedFormat.JPEG,
    '.jpeg': SupportedFormat.JPEG,
    '.png': SupportedFormat.PNG,
    '.gif': SupportedFormat.GIF,
    '.webp': SupportedFormat.WEBP,
    
    # Audio
    '.mp3': SupportedFormat.MP3,
    '.wav': SupportedFormat.WAV,
    '.ogg': SupportedFormat.OGG,
    '.m4a': SupportedFormat.M4A,
    
    # Video
    '.mp4': SupportedFormat.MP4,
    '.webm': SupportedFormat.WEBM,
    '.avi': SupportedFormat.AVI,
}

# Size limits (in bytes)
MAX_FILE_SIZES = {
    "image": 10 * 1024 * 1024,  # 10MB
    "audio": 50 * 1024 * 1024,  # 50MB  
    "video": 100 * 1024 * 1024, # 100MB
}