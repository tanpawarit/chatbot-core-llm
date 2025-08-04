# Dynamic Context Routing System

**Version**: 1.0  
**Created**: August 2025  
**Purpose**: Token optimization through intelligent context selection based on NLU analysis

## Overview

The Dynamic Context Routing System optimizes LLM token usage by selectively including only relevant context sections based on detected user intents. This results in 19-64% token savings while maintaining response quality.

## Architecture

```
User Message → NLU Analysis → Intent Detection → Context Routing → Response Generation
```

### Core Components

1. **Context Router** (`src/llm/routing.py`)
   - Analyzes NLU results to determine required contexts
   - Maps intents to specific context combinations
   - Estimates token usage and savings

2. **Response LLM** (`src/llm/node/response_llm.py`)
   - Builds system prompts with selective contexts
   - Supports backward compatibility with full contexts

3. **Configuration** (`config.yaml`)
   - Dynamic intent configuration
   - User-modifiable routing rules

## Context Types

| Context Type | Tokens | Purpose |
|--------------|--------|---------|
| `core_behavior` | 100 | Basic AI personality and behavior |
| `interaction_guidelines` | 150 | Response formatting and conversation flow |
| `product_details` | 800 | Product catalog (most expensive) |
| `business_policies` | 200 | Store policies, payment methods, services |
| `user_history` | 300 | Customer purchase history and preferences |

**Total Full Context**: 1,550 tokens

## Routing Rules

### Based on `default_intent` Configuration

```yaml
default_intent: "greet:0.3, purchase_intent:0.8, inquiry_intent:0.7, support_intent:0.6, complain_intent:0.6"
```

### Intent-to-Context Mapping

| Intent | Contexts Used | Tokens | Savings |
|--------|---------------|--------|---------|
| `greet` | core_behavior + interaction_guidelines + user_history | 550 | 64.5% |
| `purchase_intent` | core_behavior + interaction_guidelines + product_details + business_policies | 1,250 | 19.4% |
| `support_intent` | core_behavior + interaction_guidelines + business_policies + user_history | 750 | 51.6% |
| `complain_intent` | core_behavior + interaction_guidelines + business_policies + user_history | 750 | 51.6% |
| `inquiry_intent` | All contexts | 1,550 | 0% |
| `additional_intent` | All contexts | 1,550 | 0% |

### Routing Logic

```python
def determine_required_contexts(nlu_result):
    default_intents = parse_config_intents()
    detected_intents = get_intent_names(nlu_result)
    
    if "greet" in detected_intents and "greet" in default_intents:
        return minimal_contexts()
    elif "purchase_intent" in detected_intents and "purchase_intent" in default_intents:
        return product_focused_contexts()
    elif support_or_complain_detected():
        return support_focused_contexts()
    elif "inquiry_intent" in detected_intents and "inquiry_intent" in default_intents:
        return full_contexts()
    else:
        return full_contexts()  # Safe default for additional_intent
```

## Configuration Management

### Dynamic Configuration

- **Primary Source**: `config.yaml` (user-editable)
- **Environment Variables**: Only for sensitive data (API keys, Redis URL)
- **NLU Configuration**: Always loaded from YAML for maximum flexibility

### Configuration Hierarchy

1. **API Credentials**: Environment variables (secure)
2. **NLU Configuration**: YAML file (dynamic)
3. **Runtime Configuration**: Merged from both sources

### Example Configuration

```yaml
nlu:
  default_intent: "greet:0.3, purchase_intent:0.8, inquiry_intent:0.7, support_intent:0.6, complain_intent:0.6"
  additional_intent: "complaint:0.5, cancel_order:0.4, ask_price:0.6, compare_product:0.5"
  default_entity: "product, quantity, brand, price"
  additional_entity: "color, model, spec, budget, warranty, delivery"
```

## Performance Metrics

### Token Savings by Intent

| Scenario | Full Context | Optimized | Savings | Percentage |
|----------|-------------|-----------|---------|------------|
| Greeting | 1,550 | 550 | 1,000 | 64.5% |
| Product Purchase | 1,550 | 1,250 | 300 | 19.4% |
| Customer Support | 1,550 | 750 | 800 | 51.6% |
| General Inquiry | 1,550 | 1,550 | 0 | 0% |

### Real-world Impact

- **Average Savings**: ~35% across typical conversation patterns
- **Cost Reduction**: Proportional to token savings
- **Response Quality**: Maintained through selective context inclusion
- **Scalability**: Easily configurable for new intents and domains

## Usage Examples

### 1. Greeting Scenario
```
Input: "สวัสดี" (Hello)
Intent: greet
Contexts: core_behavior + interaction_guidelines + user_history
Tokens: 550 (64.5% savings)
```

### 2. Purchase Scenario
```
Input: "อยากซื้อคอมเกม" (Want to buy gaming PC)
Intent: purchase_intent
Contexts: core_behavior + interaction_guidelines + product_details + business_policies
Tokens: 1,250 (19.4% savings)
```

### 3. Support Scenario
```
Input: "ต้องการความช่วยเหลือ" (Need help)
Intent: support_intent
Contexts: core_behavior + interaction_guidelines + business_policies + user_history
Tokens: 750 (51.6% savings)
```

## Implementation Details

### Context Selection Algorithm

1. **Parse Configuration**: Extract intent lists from `config.yaml`
2. **Analyze NLU Results**: Get detected intents from user message
3. **Match Against Default Intents**: Check if detected intent is in `default_intent`
4. **Apply Routing Rules**: Select appropriate context combination
5. **Build System Prompt**: Include only selected contexts
6. **Log Performance**: Track token usage and savings

### Backward Compatibility

- **No NLU Result**: Uses full context (safe fallback)
- **Unknown Intent**: Uses full context (safe fallback)
- **Legacy Code**: Supports optional context selection parameter

### Error Handling

- **Configuration Parse Error**: Falls back to full context
- **Intent Matching Failure**: Uses full context
- **Context Building Error**: Graceful degradation with logging

## Monitoring and Logging

### Key Metrics Logged

- **Routing Decision**: Which rule was applied
- **Context Selection**: Which contexts were included/excluded
- **Token Estimation**: Estimated tokens and savings percentage
- **Intent Detection**: Detected intents and confidence scores

### Log Example

```
Context routing completed:
  routing_type=minimal
  intents=['greet']
  active_contexts=['core_behavior', 'interaction_guidelines', 'user_history']
  total_contexts=3

Token usage estimated:
  total_estimated_tokens=550
  savings_tokens=1000
  savings_percent=64.5%
  routing_type='minimal (greet)'
```

## Configuration Best Practices

### Intent Design

1. **Core Intents**: Place frequently used intents in `default_intent` for optimized routing
2. **Fallback Safety**: Keep `inquiry_intent` in `default_intent` as general-purpose fallback
3. **Specific Contexts**: Design intents to match specific context needs

### Context Optimization

1. **Expensive Contexts**: Be selective with `product_details` (800 tokens)
2. **Essential Contexts**: Always include `core_behavior` and `interaction_guidelines`
3. **Conditional Contexts**: Use `user_history` for personalization, `business_policies` for support

### Scalability Guidelines

1. **New Intents**: Add to `additional_intent` first, move to `default_intent` if routing needed
2. **New Domains**: Extend context types and routing rules as needed
3. **A/B Testing**: Monitor savings vs. response quality trade-offs

## Future Enhancements

### Planned Features

- **Entity-based Routing**: Include contexts based on detected entities
- **Confidence-based Selection**: Adjust contexts based on intent confidence scores
- **Dynamic Token Budgets**: Adaptive context selection based on available token budget
- **Performance Analytics**: Detailed metrics dashboard for optimization insights

### Extension Points

- **Custom Routing Rules**: Plugin system for domain-specific routing logic
- **Multi-language Support**: Language-specific context selection
- **User Preference Learning**: Adaptive routing based on user interaction patterns

## Troubleshooting

### Common Issues

1. **Intent Not Detected**: Check NLU configuration and training data
2. **Wrong Context**: Verify intent is in correct category (default vs. additional)
3. **No Savings**: Ensure intent mapping is correct in routing logic
4. **Configuration Not Applied**: Restart application after config changes

### Debug Commands

```bash
# Test routing logic
python -c "from src.llm.routing import context_router; print(context_router._parse_default_intents())"

# Check configuration loading
python -c "from src.config import config_manager; print(config_manager.get_config().nlu.default_intent)"
```

## Conclusion

The Dynamic Context Routing System provides significant token savings (19-64%) while maintaining response quality through intelligent context selection. The system is fully configurable via `config.yaml` and provides comprehensive logging for monitoring and optimization.

Key benefits:
- **Cost Optimization**: Substantial token savings across conversation patterns
- **Flexibility**: Easy configuration changes without code modifications  
- **Scalability**: Extensible architecture for new intents and domains
- **Reliability**: Safe fallbacks and comprehensive error handling