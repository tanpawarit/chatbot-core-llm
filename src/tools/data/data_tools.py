"""
LangChain Tools for Universal Data Querying

Simple @tool decorated functions for querying JSON data.
Business-agnostic and works with any similar data structure.
"""

from typing import List, Dict, Any, Optional
from langchain.tools import tool

from src.tools.data.json_loader import load_items
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for search (lowercase, strip whitespace)"""
    if not isinstance(text, str):
        return str(text)
    return text.lower().strip()


def _fuzzy_match(query: str, text: str, threshold: float = 0.3) -> bool:
    """
    Simple fuzzy matching for Thai and English text
    
    Args:
        query: Search query
        text: Text to search in
        threshold: Minimum match ratio (0-1)
        
    Returns:
        True if match found
    """
    query_norm = _normalize_text(query)
    text_norm = _normalize_text(text)
    
    # Exact substring match
    if query_norm in text_norm:
        return True
    
    # Word-based matching for better Thai support
    query_words = query_norm.split()
    text_words = text_norm.split()
    
    matches = 0
    for q_word in query_words:
        if any(q_word in t_word for t_word in text_words):
            matches += 1
    
    # Simple ratio matching
    ratio = matches / len(query_words) if query_words else 0
    return ratio >= threshold


@tool
def search_items_by_name(query: str) -> List[Dict[str, Any]]:
    """
    Search items by name - supports Thai and English text
    
    Args:
        query: Search term to look for in item names
        
    Returns:
        List of matching items with id, name, price, and stock
    """
    try:
        items = load_items()
        if not items:
            return []
        
        if not query or not query.strip():
            return []
        
        matching_items = []
        
        for item in items:
            name = item.get('name', '')
            if _fuzzy_match(query, name):
                matching_items.append({
                    'id': item.get('id', ''),
                    'name': name,
                    'price': item.get('price', 0),
                    'stock': item.get('stock', 0)
                })
        
        logger.info("Name search completed", 
                   query=query, 
                   results_count=len(matching_items))
        
        return matching_items
        
    except Exception as e:
        logger.error("Search by name failed", query=query, error=str(e))
        return []


@tool 
def get_item_by_id(item_id: str) -> Optional[Dict[str, Any]]:
    """
    Get specific item details by ID
    
    Args:
        item_id: Unique identifier for the item
        
    Returns:
        Item details dictionary or None if not found
    """
    try:
        items = load_items()
        if not items:
            return None
        
        if not item_id or not item_id.strip():
            return None
        
        for item in items:
            if str(item.get('id', '')).lower() == item_id.lower().strip():
                logger.info("Item found by ID", item_id=item_id)
                return item
        
        logger.info("Item not found by ID", item_id=item_id)
        return None
        
    except Exception as e:
        logger.error("Get item by ID failed", item_id=item_id, error=str(e))
        return None


@tool
def search_items_by_price_range(min_price: int, max_price: int) -> List[Dict[str, Any]]:
    """
    Find items within specified price range
    
    Args:
        min_price: Minimum price (inclusive)
        max_price: Maximum price (inclusive)
        
    Returns:
        List of items within price range
    """
    try:
        items = load_items()
        if not items:
            return []
        
        # Validate price range
        if min_price < 0 or max_price < 0 or min_price > max_price:
            return []
        
        matching_items = []
        
        for item in items:
            price = item.get('price', 0)
            if isinstance(price, (int, float)) and min_price <= price <= max_price:
                matching_items.append({
                    'id': item.get('id', ''),
                    'name': item.get('name', ''),
                    'price': price,
                    'stock': item.get('stock', 0)
                })
        
        # Sort by price
        matching_items.sort(key=lambda x: x['price'])
        
        logger.info("Price range search completed",
                   min_price=min_price,
                   max_price=max_price, 
                   results_count=len(matching_items))
        
        return matching_items
        
    except Exception as e:
        logger.error("Price range search failed", 
                    min_price=min_price, 
                    max_price=max_price,
                    error=str(e))
        return []


@tool
def check_item_stock(item_id: str) -> Dict[str, Any]:
    """
    Check stock availability for specific item
    
    Args:
        item_id: Unique identifier for the item
        
    Returns:
        Dictionary with item stock information
    """
    try:
        item = get_item_by_id.invoke({"item_id": item_id})
        
        if not item:
            return {
                'item_id': item_id,
                'found': False,
                'stock': 0,
                'availability': 'not_found'
            }
        
        stock = item.get('stock', 0)
        
        # Determine availability status
        if stock <= 0:
            availability = 'out_of_stock'
        elif stock <= 5:
            availability = 'low_stock'
        else:
            availability = 'in_stock'
        
        result = {
            'item_id': item_id,
            'name': item.get('name', ''),
            'found': True,
            'stock': stock,
            'availability': availability
        }
        
        logger.info("Stock check completed", 
                   item_id=item_id, 
                   stock=stock,
                   availability=availability)
        
        return result
        
    except Exception as e:
        logger.error("Stock check failed", item_id=item_id, error=str(e))
        return {
            'item_id': item_id,
            'found': False,
            'stock': 0,
            'availability': 'error'
        }


@tool
def get_categories() -> List[str]:
    """
    Get all unique categories from items by analyzing item names
    
    Returns:
        List of inferred categories
    """
    try:
        items = load_items()
        if not items:
            return []
        
        categories = set()
        
        # Define category keywords (can be extended)
        category_patterns = {
            'CPU': ['cpu', 'processor', 'ryzen', 'intel', 'core'],
            'VGA': ['vga', 'graphics', 'rtx', 'gtx', 'radeon'],
            'RAM': ['ram', 'memory', 'ddr4', 'ddr5'],
            'Storage': ['ssd', 'hdd', 'nvme', 'hard drive', 'storage'],
            'Monitor': ['monitor', 'จอ', 'screen', 'display'],
            'Peripherals': ['keyboard', 'mouse', 'คีย์บอร์ด', 'เมาส์'],
            'Components': ['case', 'psu', 'power', 'cooling', 'cooler', 'เคส'],
            'Audio': ['speaker', 'headset', 'ลำโพง'],
            'Gaming PC': ['gaming pc'],
            'Office PC': ['office pc'],
            'Notebook': ['notebook', 'laptop', 'โน้ตบุ๊ก']
        }
        
        for item in items:
            name = item.get('name', '').lower()
            item_categories = []
            
            # Match against category patterns
            for category, keywords in category_patterns.items():
                if any(keyword in name for keyword in keywords):
                    item_categories.append(category)
            
            # If no category matched, try to infer from common terms
            if not item_categories:
                if any(term in name for term in ['pc', 'computer']):
                    item_categories.append('Computer')
                else:
                    item_categories.append('Other')
            
            categories.update(item_categories)
        
        category_list = sorted(list(categories))
        
        logger.info("Categories extracted", 
                   categories_count=len(category_list),
                   categories=category_list)
        
        return category_list
        
    except Exception as e:
        logger.error("Get categories failed", error=str(e))
        return []