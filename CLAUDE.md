# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (fast Python package installer)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Set up environment variables (copy from example)
cp .env.example .env
# Edit .env with your API keys: OPENROUTER_API_KEY, REDIS_URL
```

### Running the Application
```bash
# Run the main chat application
python main.py

# Or with uv
uv run python main.py
```

### Development Tools
```bash
# Check syntax with pyflakes (included in dev dependencies)
python -m pyflakes src/

# Run linting manually
python -m pyflakes main.py src/
```

## Architecture Overview

This is a **conversational AI chatbot** for Thai computer sales with a **dual memory system** and **NLU-powered conversation flow**.

### Core Architecture Pattern

The system follows a **simplified workflow** that replaces LangGraph complexity:
```
A[User Message] → B{SM Valid?} → C[Load SM]/D[Load LM] → E[Create SM] → F[Save SM] → G[Add Message] 
→ H[NLU Analysis] → I{Important?} → J[Save to LM]/K[Skip] → L[Generate Response] → M[Add Response] → N[Complete]
```

### Key Components

#### 1. **Dual Memory System** (`src/memory/`)
- **Short-term Memory (SM)**: Redis-based conversation context (4-minute TTL)
- **Long-term Memory (LM)**: JSON-based customer insights and important events
- **Memory Manager**: Orchestrates the memory flow from the workflow diagram

#### 2. **NLU Processing** (`src/llm/`)
- **Classification LLM**: Analyzes messages for intent, importance, and business insights
- **Response LLM**: Generates contextual Thai responses for computer sales
- **NLU Processor**: Orchestrates the analysis and response generation flow
- **LLM Factory**: Centralized LLM instance creation with caching

#### 3. **Configuration System** (`src/config/`)
- **Environment-first**: Loads credentials from environment variables
- **YAML fallback**: Falls back to `config.yaml` for development
- **Config Manager**: Handles the dual configuration approach

#### 4. **Domain-Specific Features**
- **Thai Language Support**: All responses and error handling in Thai
- **Computer Sales Context**: Product inventory, pricing, and sales workflows
- **Business Intelligence**: Extracts customer insights from conversations

### Memory Flow Logic

The system implements a specific memory coordination pattern:

1. **SM Check**: Validate existing short-term memory in Redis
2. **Context Loading**: Load from SM (if valid) or create from LM context
3. **Message Processing**: Add user message and process through NLU
4. **Importance Assessment**: Score messages for long-term storage (threshold: 0.7)
5. **Response Generation**: Generate contextual responses with LM context
6. **State Persistence**: Save conversation state and important insights

### Configuration Priority

The system uses a **secure configuration hierarchy**:
1. **Environment Variables** (production): `OPENROUTER_API_KEY`, `REDIS_URL`
2. **YAML Configuration** (development): `config.yaml`
3. **Runtime Validation**: Validates credentials and configuration on startup

### NLU Classification System

The system classifies conversations into business-relevant categories:
- **Event Types**: INQUIRY, FEEDBACK, REQUEST, COMPLAINT, TRANSACTION, SUPPORT, INFORMATION, GENERIC_EVENT
- **Importance Scoring**: 0.0-1.0 scale for filtering important events
- **Thai Intent Recognition**: Specialized for computer sales domain

### Development Context

This is a **refactored codebase** that has undergone significant architectural improvements:
- **40% code duplication reduction** through factory patterns
- **Standardized error handling** with Thai error messages
- **Enhanced security** with environment-based configuration
- **Improved maintainability** through separation of concerns

### Product Context

The chatbot serves as a **Thai computer sales assistant** with:
- **Product Inventory**: Gaming PCs, components, peripherals with Thai pricing
- **Customer Memory**: Tracks purchase history and preferences
- **Business Intelligence**: Analyzes customer behavior and sales patterns
- **Contextual Responses**: Provides product recommendations and support

## Important Implementation Notes

- **Never hardcode API keys** - always use environment variables
- **Follow the workflow diagram** - the memory and NLU flow is specifically designed
- **Maintain Thai language support** - all user-facing text should be in Thai
- **Use the factory pattern** - LLM instances are cached and managed centrally
- **Respect the memory hierarchy** - SM for active conversations, LM for insights
- **Token tracking is automatic** - use `token_tracker.print_session_summary()` for stats