# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a proof-of-concept chatbot with a dual memory system: short-term memory (Redis-based) and long-term memory (JSON file-based). The system implements event classification using LLM to determine importance and decide what gets stored in long-term memory.

## Key Development Commands

### Running the Application
```bash
python main.py
```

### Dependencies Management
```bash
# Install dependencies (using uv)
uv sync

# Add new dependency
uv add <package-name>
```

### Configuration
- Main config file: `config.yaml` (contains API keys and model settings)
- Configuration models defined in `src/models.py`
- Redis connection and LLM endpoints configured via `config.yaml`

## Architecture Overview

### Memory Flow Architecture
The system follows a specific flow diagram (A→B→C→...→G):

1. **User Message Processing** (A→G):
   - Check if Short-term Memory (SM) exists and is valid in Redis
   - If not valid: Load Long-term Memory (LM) from JSON → Create SM from LM context
   - Add user message to SM and save to Redis

2. **Event Processing** (H→K):
   - LLM classifies user message into event types with importance scores
   - If importance ≥ 0.7: Save event to Long-term Memory (JSON file)
   - Generate chat response using separate LLM

3. **Response Handling** (M→N):
   - Generate assistant response using conversation context
   - Add response to SM in Redis

### Core Components

#### Memory System (`src/memory/`)
- **MemoryManager**: Orchestrates the entire memory flow
- **ShortTermMemory**: Redis-based storage with TTL (30 min default)
- **LongTermMemory**: JSON file-based persistent storage in `data/longterm/`

#### LLM Processing (`src/llm/`)
- **EventProcessor**: Handles event classification and response generation
- **Two separate LLMs**: 
  - Classification LLM (gpt-4o-mini, temperature 0.1) for structured JSON output
  - Response LLM (gpt-4, temperature 0.7) for natural conversation

#### Data Models (`src/models.py`)
- **Message/Conversation**: Chat message handling with roles and timestamps
- **Event/EventClassification**: Event types (INQUIRY, FEEDBACK, REQUEST, etc.) with importance scoring
- **LongTermMemory**: Persistent storage for important events
- **Config models**: Pydantic models for configuration validation

### Global Instances
The system uses global singleton instances:
- `memory_manager` in `src/memory/manager.py`
- `event_processor` in `src/llm/processor.py`
- `short_term_memory` in `src/memory/short_term.py`
- `long_term_memory` in `src/memory/long_term.py`

## Key Design Patterns

### Event Classification System
- Uses structured prompts with specific event types and importance scales
- JSON-only output for classification with importance scores 0.0-1.0
- Threshold-based filtering (≥0.7) for long-term memory storage

### Dual LLM Architecture
- Separate specialized LLMs for different tasks:
  - Classification: Lower temperature, shorter responses, structured output
  - Response: Higher temperature, longer responses, conversational

### Memory Hierarchy
- Redis (SM): Fast, temporary, conversation context with TTL
- JSON Files (LM): Persistent, important events only, file per conversation

### Configuration Management
- Centralized YAML configuration with Pydantic validation
- Separate model configs for classification vs response generation
- Environment-specific settings (Redis URLs, file paths)

## Dependencies
- **LangChain**: LLM orchestration and OpenRouter integration
- **Redis**: Short-term memory storage with TTL support
- **Pydantic**: Data validation and serialization
- **structlog**: Structured logging throughout the system

## Important Files
- `main.py`: Entry point with chat interface and debug output
- `config.yaml`: API keys, model settings, Redis connection (sensitive)
- `src/llm/prompt.py`: Event classification system prompt
- `data/longterm/`: Directory for persistent JSON memory files