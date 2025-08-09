# Tools Architecture

Scalable LangChain tools system with automatic discovery and category management.

## Architecture Overview

```
src/tools/
├── __init__.py           # Main exports and tool loading
├── base_tool.py          # Base classes and registry
├── tool_loader.py        # Auto-discovery system
├── json_loader.py        # JSON data utilities
├── data_tools.py         # Data querying tools
├── qr_tools.py           # QR code tools (placeholder)
└── README.md            # This file
```

## Core Components

### 1. Tool Registry (`base_tool.py`)
- **ToolRegistry**: Central registry for all tools
- **ToolCategory**: Predefined categories (data, utility, integration, etc.)
- **BaseToolConfig**: Configuration model for tools
- **ScalableToolMixin**: Common functionality for tools

### 2. Auto-Discovery (`tool_loader.py`)
- **ToolLoader**: Automatically discovers `*_tools.py` files
- **Dynamic Loading**: Imports and registers tools at runtime
- **Category Inference**: Maps modules to categories automatically

### 3. Tool Categories
- **DATA**: Data querying and manipulation tools
- **UTILITY**: General utility tools (QR codes, converters)
- **INTEGRATION**: External API and service integration
- **COMMUNICATION**: Messaging and notification tools
- **AUTOMATION**: Workflow and process automation

## Usage Examples

### Current Usage (Backward Compatible)
```python
from src.tools import AVAILABLE_TOOLS

# Use with LangChain LLM
llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)
```

### New Scalable Usage
```python
from src.tools import get_available_tools, tool_registry, ToolCategory

# Auto-discover all tools
all_tools = get_available_tools()

# Get tools by category
data_tools = tool_registry.get_tools_by_category(ToolCategory.DATA)
utility_tools = tool_registry.get_tools_by_category(ToolCategory.UTILITY)

# Use specific tool categories
llm_with_data_tools = llm.bind_tools(data_tools)
```

## Adding New Tools

### Step 1: Create Tool Module
Create `src/tools/your_category_tools.py`:

```python
from langchain.tools import tool
from .base_tool import ScalableToolMixin

@tool
def your_new_tool(param: str) -> dict:
    """Your tool description"""
    # Implementation
    return {"result": "success"}
```

### Step 2: Auto-Discovery
Tools are automatically discovered and loaded:
- Module name pattern: `*_tools.py`
- Category inferred from module name
- Tools registered automatically

### Step 3: Usage
```python
# Tools are automatically available
from src.tools import get_available_tools
tools = get_available_tools()  # Includes your new tool
```

## Configuration

Tools can be configured through:
- Environment variables
- YAML configuration
- Runtime parameters

## Future Extensions

### Planned Tool Categories
- **notification_tools.py**: Email, SMS, push notifications
- **api_tools.py**: REST API clients, webhooks
- **file_tools.py**: File operations, conversions
- **automation_tools.py**: Workflow automation
- **integration_tools.py**: Third-party service integration

### Advanced Features
- Tool versioning
- Rate limiting
- Caching
- Error recovery
- Monitoring and analytics