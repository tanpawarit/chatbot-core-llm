"""
Scalable LangChain Tools Package

Organized structure:
- data/: Data querying and manipulation tools
- utility/: General utility tools (QR codes, converters, etc.)
- base_tool.py & tool_loader.py: Core infrastructure at root level

Features:
- Automatic tool discovery and loading
- Category-based organization
- Registry pattern for tool management
- Easy extension for new tool types
"""

# Import from organized subfolders
from src.tools.data import (
    search_items_by_name,
    get_item_by_id,
    search_items_by_price_range,
    check_item_stock,
    get_categories,
)

from src.tools.base_tool import tool_registry, ToolCategory
from src.tools.tool_loader import tool_loader

# Auto-load all available tools
def get_available_tools():
    """Get all available tools (auto-discovered)"""
    return tool_loader.auto_load_all_tools()

# Main tool list using current data tools
AVAILABLE_TOOLS = [
    search_items_by_name,
    get_item_by_id,
    search_items_by_price_range, 
    check_item_stock,
    get_categories,
]

# Export everything for easy access
__all__ = [
    # Data tools
    "search_items_by_name",
    "get_item_by_id", 
    "search_items_by_price_range",
    "check_item_stock",
    "get_categories",
    
    # Tool infrastructure
    "tool_registry",
    "tool_loader", 
    "ToolCategory",
    "get_available_tools",
    "AVAILABLE_TOOLS",
]