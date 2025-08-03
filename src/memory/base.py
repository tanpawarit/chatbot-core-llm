"""
Base classes for memory operations
Provides common interface and error handling patterns
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryOperationResult:
    """Standard result object for memory operations"""
    
    def __init__(self, success: bool, data: Optional[Any] = None, error: Optional[str] = None, metadata: Optional[Dict] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def __bool__(self):
        return self.success
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"MemoryOperationResult({status}, error='{self.error}', metadata={self.metadata})"


class BaseMemoryStore(ABC):
    """
    Abstract base class for memory storage implementations.
    Provides common interface and error handling patterns.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage"""
        pass
    
    @abstractmethod
    def save(self, key: str, data: Any, **kwargs) -> MemoryOperationResult:
        """Save data to storage"""
        pass
    
    @abstractmethod
    def load(self, key: str, **kwargs) -> MemoryOperationResult:
        """Load data from storage"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> MemoryOperationResult:
        """Delete data from storage"""
        pass
    
    def _log_operation(self, operation: str, key: str, success: bool, error: Optional[str] = None, **metadata):
        """Standardized logging for memory operations"""
        if success:
            self.logger.info(f"{operation} successful", 
                           storage=self.name,
                           key=key,
                           **metadata)
        else:
            self.logger.error(f"{operation} failed", 
                            storage=self.name,
                            key=key,
                            error=error,
                            **metadata)
    
    def _create_result(self, success: bool, data: Optional[Any] = None, error: Optional[str] = None, **metadata) -> MemoryOperationResult:
        """Create standardized result object"""
        return MemoryOperationResult(success, data, error, metadata)


class BaseTemporalMemory(BaseMemoryStore):
    """
    Base class for temporal memory (with TTL support).
    Extended by ShortTermMemory implementations.
    """
    
    @abstractmethod
    def is_valid(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        pass
    
    @abstractmethod
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        pass
    
    @abstractmethod
    def set_ttl(self, key: str, ttl: int) -> MemoryOperationResult:
        """Set TTL for existing key"""
        pass


class BasePersistentMemory(BaseMemoryStore):
    """
    Base class for persistent memory (file-based storage).
    Extended by LongTermMemory implementations.
    """
    
    @abstractmethod
    def backup(self, key: str, backup_path: Optional[str] = None) -> MemoryOperationResult:
        """Create backup of data"""
        pass
    
    @abstractmethod
    def restore(self, key: str, backup_path: str) -> MemoryOperationResult:
        """Restore data from backup"""
        pass
    
    @abstractmethod
    def get_file_info(self, key: str) -> Optional[Dict]:
        """Get file information (size, modified date, etc.)"""
        pass


class MemoryCoordinator:
    """
    Coordinates operations between different memory stores.
    Handles fallback strategies and error recovery.
    """
    
    def __init__(self):
        self.stores: Dict[str, BaseMemoryStore] = {}
        self.logger = get_logger(f"{__name__}.coordinator")
    
    def register_store(self, name: str, store: BaseMemoryStore):
        """Register a memory store"""
        self.stores[name] = store
        self.logger.info("Memory store registered", name=name, type=type(store).__name__)
    
    def get_store(self, name: str) -> Optional[BaseMemoryStore]:
        """Get registered memory store"""
        return self.stores.get(name)
    
    def execute_with_fallback(self, primary_store: str, fallback_store: str, operation: str, key: str, **kwargs) -> MemoryOperationResult:
        """
        Execute operation with fallback strategy
        
        Args:
            primary_store: Primary store name
            fallback_store: Fallback store name  
            operation: Operation to execute ('save', 'load', 'delete')
            key: Key for operation
            **kwargs: Additional arguments for operation
        """
        primary = self.get_store(primary_store)
        fallback = self.get_store(fallback_store)
        
        if not primary:
            return MemoryOperationResult(False, error=f"Primary store '{primary_store}' not found")
        
        try:
            # Try primary store
            if hasattr(primary, operation):
                result = getattr(primary, operation)(key, **kwargs)
                if result.success:
                    return result
                else:
                    self.logger.warning("Primary store operation failed, trying fallback",
                                      primary=primary_store,
                                      fallback=fallback_store,
                                      operation=operation,
                                      error=result.error)
        except Exception as e:
            self.logger.error("Primary store operation error",
                            primary=primary_store,
                            operation=operation,
                            error=str(e))
        
        # Try fallback store
        if fallback and hasattr(fallback, operation):
            try:
                result = getattr(fallback, operation)(key, **kwargs)
                result.metadata["used_fallback"] = True
                result.metadata["primary_store"] = primary_store
                result.metadata["fallback_store"] = fallback_store
                return result
            except Exception as e:
                self.logger.error("Fallback store operation error",
                                fallback=fallback_store,
                                operation=operation,
                                error=str(e))
        
        return MemoryOperationResult(False, error="Both primary and fallback operations failed")


# Global coordinator instance
memory_coordinator = MemoryCoordinator()