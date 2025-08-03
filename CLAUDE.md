# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a proof-of-concept chatbot system implementing a dual memory architecture with Thai computer sales assistant functionality. The system uses two LLMs for event classification and response generation, backed by Redis for short-term memory and JSON files for long-term memory.

## Core Architecture

### Memory System
- **Short-term Memory (SM)**: Redis-based conversation storage with TTL
- **Long-term Memory (LM)**: JSON file-based persistent storage for important events
- **Event Classification**: LLM determines event importance (0.0-1.0 scale)
- **Threshold**: Events ≥0.7 importance are saved to long-term memory

### Processing Flow
The system implements this workflow (as shown in README.md mermaid diagram):
1. Check if short-term memory exists → Load SM or create from LM
2. Add user message to conversation
3. Classify event using LLM (returns EventType and importance score)
4. If important (≥0.7) → Save to long-term memory
5. Generate response using conversation + LM context
6. Add response to conversation

### Key Components
- `main.py`: CLI interface and simplified workflow orchestration
- `src/models.py`: Pydantic models for Message, Event, Conversation, etc.
- `src/llm/processor.py`: Event processing orchestrator
- `src/memory/manager.py`: Memory flow coordination
- `src/llm/node/`: Individual LLM nodes for classification and response
- `src/memory/`: Short-term (Redis) and long-term (JSON) memory implementations

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Or install manually
pip install -r requirements.txt  # if requirements.txt exists
```

### Running the Application
```bash
# Run the chatbot CLI
python main.py

# Run with uv
uv run python main.py
```

### Configuration
- Main config: `config.yaml` (contains OpenRouter API keys and Redis credentials)
- Models use Gemini 2.5 Flash Lite via OpenRouter
- Redis instance hosted on Upstash

### Memory Data
- Long-term memory files: `data/longterm/[user_id].json`
- Product catalog: `data/product_detail/products.json`
- Sample user data available in `data/longterm/`

## Code Patterns

### Message Flow
All messages use the `Message` model with `MessageRole` enum (USER/ASSISTANT/SYSTEM). The `Conversation` class manages message lists with timestamps.

### Event Classification
Events are classified into 8 types: INQUIRY, FEEDBACK, REQUEST, COMPLAINT, TRANSACTION, SUPPORT, INFORMATION, GENERIC_EVENT. Each gets an importance score that determines LM persistence.

### Error Handling
- LLM failures return fallback Thai error messages
- Memory operations include existence/validity checks
- Comprehensive logging using structlog

### Dependencies
- LangChain ecosystem for LLM operations
- Pydantic for data validation
- Redis for session storage
- Structured logging with structlog

## Thai Language Context

The chatbot acts as a Thai computer sales assistant with:
- Product catalog with Thai descriptions and pricing
- Thai language responses and error messages
- Cultural context for computer sales in Thailand
- Sample conversations showing typical customer interactions

## Important Notes

- API keys are currently exposed in `config.yaml` - should be moved to environment variables
- Redis URL contains credentials - should use secure credential management
- The system implements a simplified alternative to LangGraph complexity
- All timestamps use UTC timezone