import redis
from typing import Optional, Any
import json
from src.config import config_manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._config = config_manager.get_memory_config()
    
    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self._config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                self._client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error("Failed to connect to Redis", error=str(e))
                raise
        return self._client
    
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            json_data = json.dumps(value, default=str)
            if ttl:
                return self.client.setex(key, ttl, json_data)
            else:
                return self.client.set(key, json_data)
        except Exception as e:
            logger.error("Failed to set JSON data", key=key, error=str(e))
            return False
    
    def get_json(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("Failed to get JSON data", key=key, error=str(e))
            return None
    
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error("Failed to check key existence", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error("Failed to delete key", key=key, error=str(e))
            return False
    
    def get_ttl(self, key: str) -> int:
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error("Failed to get TTL", key=key, error=str(e))
            return -1


# Global Redis client instance
redis_client = RedisClient()