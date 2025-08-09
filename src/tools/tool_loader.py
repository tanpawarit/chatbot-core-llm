"""
Dynamic Tool Loader

Automatically discovers and loads tools from tool modules
"""

import importlib
import inspect
from typing import List, Dict, Any
from pathlib import Path
from langchain.tools import BaseTool

from src.tools.base_tool import tool_registry, ToolCategory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ToolLoader:
    """Dynamic tool discovery and loading"""
    
    def __init__(self, tools_package: str = "src.tools"):
        self.tools_package = tools_package
        self.loaded_modules: Dict[str, Any] = {}
    
    def discover_tool_modules(self) -> List[str]:
        """Discover all tool modules in the tools package (including subfolders)"""
        tools_path = Path(__file__).parent
        tool_modules = []
        
        # Search in root tools directory
        for file_path in tools_path.glob("*_tools.py"):
            if file_path.name != "__init__.py":
                module_name = file_path.stem
                tool_modules.append(f"{self.tools_package}.{module_name}")
        
        # Search in subfolders
        for subfolder in tools_path.iterdir():
            if subfolder.is_dir() and not subfolder.name.startswith('_'):
                for file_path in subfolder.glob("*_tools.py"):
                    if file_path.name != "__init__.py":
                        module_name = file_path.stem
                        tool_modules.append(f"{self.tools_package}.{subfolder.name}.{module_name}")
                
        logger.info(f"Discovered tool modules: {tool_modules}")
        return tool_modules
    
    def load_module_tools(self, module_path: str) -> List[BaseTool]:
        """Load all tools from a specific module"""
        try:
            module = importlib.import_module(module_path)
            self.loaded_modules[module_path] = module
            
            tools = []
            for name, obj in inspect.getmembers(module):
                # Look for LangChain tool functions
                if hasattr(obj, '_langchain_tool') or isinstance(obj, BaseTool):
                    tools.append(obj)
                    logger.info(f"Loaded tool: {name} from {module_path}")
            
            return tools
            
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading tools from {module_path}: {e}")
            return []
    
    def auto_load_all_tools(self) -> List[BaseTool]:
        """Automatically discover and load all tools"""
        all_tools = []
        
        # Discover tool modules
        tool_modules = self.discover_tool_modules()
        
        # Load tools from each module
        for module_path in tool_modules:
            module_tools = self.load_module_tools(module_path)
            all_tools.extend(module_tools)
            
            # Register tools in registry by category
            category = self._infer_category_from_module(module_path)
            for tool in module_tools:
                tool_registry.register_tool(tool, category)
        
        logger.info(f"Auto-loaded {len(all_tools)} tools total")
        return all_tools
    
    def _infer_category_from_module(self, module_path: str) -> str:
        """Infer tool category from module name"""
        module_name = module_path.split('.')[-1]
        
        category_mapping = {
            'data_tools': ToolCategory.DATA,
            'qr_tools': ToolCategory.UTILITY,
            'api_tools': ToolCategory.INTEGRATION,
            'notification_tools': ToolCategory.COMMUNICATION,
            'automation_tools': ToolCategory.AUTOMATION,
        }
        
        return category_mapping.get(module_name, ToolCategory.UTILITY)
    
    def reload_tools(self) -> List[BaseTool]:
        """Reload all tools (useful for development)"""
        # Clear loaded modules
        for module_path in self.loaded_modules:
            if module_path in importlib.sys.modules:
                importlib.reload(importlib.sys.modules[module_path])
        
        # Reload all tools
        return self.auto_load_all_tools()


# Global tool loader instance
tool_loader = ToolLoader()