"""
Environment variable loader with fallback support
Handles loading configuration from environment variables with YAML fallback
"""

import os
from typing import Optional, Dict, overload
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnvLoader:
    """
    Environment variable loader with validation and fallback support.
    Supports loading from .env files and environment variables.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self._load_env_file()
    
    def _load_env_file(self):
        """Load environment variables from .env file if exists"""
        env_path = Path(self.env_file)
        if env_path.exists():
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and not os.getenv(key):  # Don't override existing env vars
                                os.environ[key] = value
                
                logger.info("Environment file loaded", path=str(env_path))
            except Exception as e:
                logger.warning("Failed to load environment file", path=str(env_path), error=str(e))
        else:
            logger.debug("Environment file not found", path=str(env_path))
    
    @overload
    def get_str(self, key: str, default: str, required: bool = False) -> str: ...
    
    @overload
    def get_str(self, key: str, *, required: bool = True) -> str: ...
    
    @overload
    def get_str(self, key: str, default: None = None, required: bool = False) -> Optional[str]: ...
    
    def get_str(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Get string value from environment
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the value is required (raises error if missing)
            
        Returns:
            String value or None/default
        """
        value = os.getenv(key, default)
        
        if required and not value:
            raise ValueError(f"Required environment variable '{key}' is not set")
        
        return value
    
    @overload
    def get_int(self, key: str, default: int, required: bool = False) -> int: ...
    
    @overload
    def get_int(self, key: str, *, required: bool = True) -> int: ...
    
    @overload
    def get_int(self, key: str, default: None = None, required: bool = False) -> Optional[int]: ...
    
    def get_int(self, key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
        """Get integer value from environment"""
        value = self.get_str(key, required=required)
        
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            if required:
                raise ValueError(f"Environment variable '{key}' must be an integer, got: {value}")
            logger.warning("Invalid integer value in environment", key=key, value=value, using_default=default)
            return default
    
    @overload
    def get_float(self, key: str, default: float, required: bool = False) -> float: ...
    
    @overload
    def get_float(self, key: str, *, required: bool = True) -> float: ...
    
    @overload
    def get_float(self, key: str, default: None = None, required: bool = False) -> Optional[float]: ...
    
    def get_float(self, key: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
        """Get float value from environment"""
        value = self.get_str(key, required=required)
        
        if value is None:
            return default
        
        try:
            return float(value)
        except ValueError:
            if required:
                raise ValueError(f"Environment variable '{key}' must be a float, got: {value}")
            logger.warning("Invalid float value in environment", key=key, value=value, using_default=default)
            return default
    
    @overload
    def get_bool(self, key: str, default: bool, required: bool = False) -> bool: ...
    
    @overload
    def get_bool(self, key: str, *, required: bool = True) -> bool: ...
    
    @overload
    def get_bool(self, key: str, default: None = None, required: bool = False) -> Optional[bool]: ...
    
    def get_bool(self, key: str, default: Optional[bool] = None, required: bool = False) -> Optional[bool]:
        """Get boolean value from environment"""
        value = self.get_str(key, required=required)
        
        if value is None:
            return default
        
        # Convert string to boolean
        if value.lower() in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off', 'disabled'):
            return False
        else:
            if required:
                raise ValueError(f"Environment variable '{key}' must be a boolean, got: {value}")
            logger.warning("Invalid boolean value in environment", key=key, value=value, using_default=default)
            return default
    
    def has_credential(self, key: str) -> bool:
        """Check if credential exists without logging the value"""
        value = os.getenv(key)
        has_value = bool(value and value.strip())
        logger.debug("Credential check", key=key, has_value=has_value)
        return has_value
    
    def validate_required_credentials(self, required_keys: list[str]) -> Dict[str, bool]:
        """
        Validate that all required credentials are present
        
        Args:
            required_keys: List of required environment variable names
            
        Returns:
            Dict mapping key to whether it exists
        """
        validation_results = {}
        missing_keys = []
        
        for key in required_keys:
            has_value = self.has_credential(key)
            validation_results[key] = has_value
            if not has_value:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error("Missing required credentials", missing_keys=missing_keys)
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        logger.info("All required credentials validated", keys=required_keys)
        return validation_results


# Global env loader instance
env_loader = EnvLoader()