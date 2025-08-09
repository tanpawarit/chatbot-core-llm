"""
Simple JSON Data Loader for LangChain Tools

Provides utilities for loading and caching JSON data from files.
Business-agnostic and easily configurable.
"""

import json
import os
from typing import Dict, List, Any, Optional
from functools import lru_cache

from src.utils.logging import get_logger

logger = get_logger(__name__)


class JSONDataLoader:
    """Simple JSON data loader with caching and error handling"""
    
    def __init__(self, file_path: str = "data/product_detail/products.json"):
        """
        Initialize loader with file path
        
        Args:
            file_path: Path to JSON data file
        """
        self.file_path = file_path
        self._cache = None
        self._cache_timestamp = None
    
    def load_data(self, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load JSON data with caching
        
        Args:
            force_reload: Force reload from file even if cached
            
        Returns:
            Loaded data dictionary or None if error
        """
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                logger.warning("Data file not found", path=self.file_path)
                return None
            
            # Get file modification time
            file_mtime = os.path.getmtime(self.file_path)
            
            # Use cache if available and file hasn't changed
            if (not force_reload and 
                self._cache is not None and 
                self._cache_timestamp == file_mtime):
                return self._cache
            
            # Load from file
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update cache
            self._cache = data
            self._cache_timestamp = file_mtime
            
            logger.info("Data loaded successfully", 
                       path=self.file_path,
                       items_count=len(data.get('products', data.get('items', []))))
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON format", path=self.file_path, error=str(e))
            return None
        except Exception as e:
            logger.error("Failed to load data", path=self.file_path, error=str(e))
            return None
    
    def get_items(self) -> List[Dict[str, Any]]:
        """
        Get items array from loaded data
        
        Supports multiple data structures:
        - {"products": [...]}  (current format)
        - {"items": [...]}     (generic format)
        - [...]                (direct array)
        
        Returns:
            List of item dictionaries
        """
        data = self.load_data()
        if not data:
            return []
        
        # Handle different data structures
        if isinstance(data, list):
            return data
        elif 'products' in data:
            return data['products']
        elif 'items' in data:
            return data['items']
        else:
            logger.warning("Unknown data structure", keys=list(data.keys()))
            return []
    
    def clear_cache(self):
        """Clear cached data"""
        self._cache = None
        self._cache_timestamp = None


# Global loader instance
default_loader = JSONDataLoader()


# Convenience functions
@lru_cache(maxsize=1)
def get_default_loader() -> JSONDataLoader:
    """Get the default data loader instance"""
    return default_loader


def load_items(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load items from JSON file
    
    Args:
        file_path: Optional custom file path
        
    Returns:
        List of item dictionaries
    """
    if file_path:
        loader = JSONDataLoader(file_path)
        return loader.get_items()
    else:
        return default_loader.get_items()


def reload_data():
    """Force reload data from file"""
    default_loader.load_data(force_reload=True)