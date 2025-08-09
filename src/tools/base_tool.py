"""
Base Tool Classes and Utilities

Provides base classes and utilities for building scalable LangChain tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ToolCategory:
    """Tool category constants"""
    DATA = "data"
    UTILITY = "utility"
    INTEGRATION = "integration"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"


class BaseToolConfig(BaseModel):
    """Base configuration for tools"""
    name: str
    description: str
    category: str
    enabled: bool = True
    rate_limit: Optional[int] = None
    timeout: Optional[int] = None


class ScalableToolMixin:
    """Mixin providing common tool functionality"""
    
    def log_tool_usage(self, tool_name: str, args: Dict[str, Any], result: Any):
        """Log tool usage for monitoring"""
        logger.info(f"Tool executed: {tool_name}", 
                   args=args, 
                   result_type=type(result).__name__)
    
    def handle_tool_error(self, tool_name: str, error: Exception) -> Dict[str, Any]:
        """Standard error handling for tools"""
        error_msg = f"Tool {tool_name} failed: {str(error)}"
        logger.error(error_msg, error=str(error))
        return {
            'success': False,
            'error': error_msg,
            'result': None
        }


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: BaseTool, category: str):
        """Register a tool in the registry"""
        self._tools[tool.name] = tool
        
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} in category: {category}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self._tools.values())
    
    def list_categories(self) -> List[str]:
        """List all available categories"""
        return list(self._categories.keys())


# Global tool registry instance
tool_registry = ToolRegistry()