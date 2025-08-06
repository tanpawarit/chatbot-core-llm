"""Simple media processor for validation and handling"""

import mimetypes
from pathlib import Path

from src.media.types import MediaInfo, SupportedFormat, EXTENSION_MAP, MAX_FILE_SIZES
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MediaProcessor:
    """Simple media file processor and validator"""
    
    def __init__(self):
        # Using mimetypes for MIME type detection
        pass
    
    def validate_file(self, file_path: Path) -> MediaInfo:
        """Validate a media file and return MediaInfo"""
        try:
            if not file_path.exists():
                return MediaInfo(
                    file_path=file_path,
                    media_type="unknown",
                    format=SupportedFormat.JPEG,  # dummy
                    file_size=0,
                    is_valid=False,
                    error_message="File not found"
                )
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Get format from extension
            extension = file_path.suffix.lower()
            if extension not in EXTENSION_MAP:
                return MediaInfo(
                    file_path=file_path,
                    media_type="unknown", 
                    format=SupportedFormat.JPEG,  # dummy
                    file_size=file_size,
                    is_valid=False,
                    error_message=f"Unsupported file format: {extension}"
                )
            
            format_type = EXTENSION_MAP[extension]
            media_type = self._get_media_type(format_type)
            
            # Validate file size
            max_size = MAX_FILE_SIZES.get(media_type, MAX_FILE_SIZES["image"])
            if file_size > max_size:
                return MediaInfo(
                    file_path=file_path,
                    media_type=media_type,
                    format=format_type,
                    file_size=file_size,
                    is_valid=False,
                    error_message=f"File too large: {file_size/1024/1024:.1f}MB > {max_size/1024/1024}MB"
                )
            
            # Validate MIME type using mimetypes
            guessed_type, _ = mimetypes.guess_type(str(file_path.resolve()))
            if guessed_type and not self._mime_matches_format(guessed_type, format_type):
                return MediaInfo(
                    file_path=file_path,
                    media_type=media_type,
                    format=format_type,
                    file_size=file_size,
                    is_valid=False,
                    error_message=f"File type mismatch: {guessed_type} vs {format_type.value}"
                )
            
            # All validations passed
            return MediaInfo(
                file_path=file_path,
                media_type=media_type,
                format=format_type,
                file_size=file_size,
                is_valid=True
            )
            
        except Exception as e:
            logger.error("File validation error", file_path=str(file_path), error=str(e))
            return MediaInfo(
                file_path=file_path,
                media_type="unknown",
                format=SupportedFormat.JPEG,  # dummy
                file_size=0,
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def _get_media_type(self, format_type: SupportedFormat) -> str:
        """Get media type category from format"""
        format_value = format_type.value
        if format_value.startswith("image/"):
            return "image"
        elif format_value.startswith("audio/"):
            return "audio"
        elif format_value.startswith("video/"):
            return "video"
        else:
            return "unknown"
    
    def _mime_matches_format(self, detected_mime: str, expected_format: SupportedFormat) -> bool:
        """Check if detected MIME type matches expected format"""
        expected_mime = expected_format.value
        
        # Handle common MIME variations
        mime_variants = {
            "image/jpeg": ["image/jpeg", "image/jpg"],
            "audio/mpeg": ["audio/mpeg", "audio/mp3"],
            "audio/mp4": ["audio/mp4", "audio/m4a"],
        }
        
        expected_variants = mime_variants.get(expected_mime, [expected_mime])
        return detected_mime in expected_variants
    
    def get_media_summary(self, media_info: MediaInfo) -> str:
        """Get human-readable summary of media file"""
        if not media_info.is_valid:
            return f"❌ Invalid file: {media_info.error_message}"
        
        return f"📎 {media_info.media_type.title()} file: {media_info.file_path.name} ({media_info.size_mb:.1f}MB)"