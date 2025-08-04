# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Project Setup
```bash
# Install dependencies using uv
uv sync

# Set up environment variables (copy from .env.example)
cp .env.example .env
# Edit .env with your OpenRouter API key and Redis URL
```

### Running the Application
```bash
# Run the main chat interface
python main.py

# Run with uv
uv run python main.py
```

### Code Quality
```bash
# Run linting (using pyflakes)
uv run pyflakes src/

# Check dependencies
uv tree
```

## High-Level Architecture

### Core System Flow
The application implements a dual-memory chatbot system with intelligent context routing:

```
User Input → Memory Manager → LLM Processor → Response Generation
     ↓              ↓              ↓               ↓
   Message      SM/LM Flow    NLU Analysis   Context Routing
```

### Key Components

#### 1. Memory System (`src/memory/`)
- **Short-term Memory (SM)**: Redis-based conversation storage with 4-minute TTL
- **Long-term Memory (LM)**: JSON file storage for important user analyses 
- **Memory Manager**: Orchestrates SM/LM flow as per diagram in README.md
- **Flow**: User message → Check SM validity → Load/Create SM → Add to SM → Process → Save important NLU to LM

#### 2. LLM Processing (`src/llm/`)
- **Processor**: Main orchestrator handling NLU analysis and response generation
- **Factory Pattern**: Centralized LLM instance creation with caching (classification + response models)
- **Dynamic Context Routing**: Token optimization system (19-64% savings) based on detected intents
- **NLU Nodes**: Separate LLMs for classification vs response generation

#### 3. Configuration System (`src/config/`)
- **Hybrid Configuration**: Environment variables (sensitive) + YAML config (non-sensitive)  
- **Manager**: Centralized config access with validation
- **Env Loader**: Secure credential loading with fallbacks

#### 4. Utilities (`src/utils/`)
- **Token Tracker**: Session-wide usage and cost monitoring
- **Redis Client**: Connection management with error handling
- **Logging**: Structured logging with Thai language support
- **Cost Calculator**: LLM usage cost tracking

### Architecture Principles

#### Dynamic Context Routing System
The system implements intelligent token optimization through intent-based context selection:

- **Intent Detection**: NLU analysis determines user intents (greet, purchase_intent, support_intent, etc.)
- **Context Mapping**: Different intents trigger different context combinations:
  - `greet`: minimal contexts (550 tokens, 64.5% savings)  
  - `purchase_intent`: product-focused contexts (1,250 tokens, 19.4% savings)
  - `support_intent`: support-focused contexts (750 tokens, 51.6% savings)
- **Configuration-Driven**: Intent routing rules defined in `config.yaml`
- **Fallback Safety**: Unknown intents use full context (1,550 tokens)

#### Dual Memory Architecture
- **SM (Short-term)**: Active conversation state in Redis, 4-minute TTL
- **LM (Long-term)**: Important analyses saved to JSON files based on importance threshold (≥0.7)
- **Importance Scoring**: Multi-factor scoring considering message length, intent specificity, and entity presence
- **Context Coordination**: Response generation uses both SM conversation history and LM user insights

#### LLM Usage Patterns  
- **Classification LLM**: google/gemini-2.5-flash-lite for NLU analysis (JSON output, low temperature)
- **Response LLM**: google/gemini-2.5-flash-lite for chat responses (text output, higher temperature)
- **Factory Pattern**: Cached instances, centralized configuration
- **Token Tracking**: Automatic usage monitoring with session statistics

### Key Files to Understand

#### Core Flow Implementation
- `main.py`: Main chat interface implementing A→B→C...→N workflow from README diagram
- `src/memory/manager.py`: Memory orchestration following the flowchart logic
- `src/llm/processor.py`: NLU analysis and response generation pipeline

#### Architecture Components  
- `src/llm/routing.py`: Dynamic context routing logic and token optimization
- `src/llm/factory.py`: LLM instance management with caching
- `src/config/manager.py`: Centralized configuration with env/YAML hybrid loading
- `src/models.py`: Core data models (Message, Conversation, NLUResult)

#### Configuration Files
- `config.yaml`: NLU intent/entity configuration, LLM settings, memory TTL
- `.env`: API credentials (OpenRouter key, Redis URL) - use `.env.example` as template

### Development Context

#### Recent Refactoring (see note/REFACTORING_SUMMARY.md)
- Implemented factory patterns to eliminate code duplication (~40% reduction)
- Centralized token tracking and cost calculation
- Enhanced security with environment-first configuration  
- Standardized error handling with Thai language support
- Pipeline pattern for modular message processing

#### Domain Context
- **Target**: Thai computer sales chatbot
- **NLU Intents**: greet, purchase_intent, inquiry_intent, support_intent, complain_intent
- **Entity Types**: product, quantity, brand, price, color, model, spec, budget
- **Language**: Thai language support throughout (error messages, responses)

#### Memory & Performance
- **Token Budget**: 500-2000 tokens per response, optimized via context routing
- **Memory TTL**: 4-minute conversation expiry in Redis
- **Cost Tracking**: Automatic token usage and cost monitoring per session  
- **Caching**: LLM instance caching, conversation state caching