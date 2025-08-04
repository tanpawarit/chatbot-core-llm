"""Simple cost calculation utilities for LLM usage""" 

# Approximate pricing for common models (per 1M tokens)
# These are rough estimates - check current pricing from providers
MODEL_PRICING = {
    # Google models (OpenRouter pricing)
    "google/gemini-2.5-flash-lite": {
        "input": 0.1,   # $0.075 per 1M input tokens
        "output": 0.4,   # $0.30 per 1M output tokens
    },
    "anthropic/claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
    },
    # OpenAI models (approximate)
    "openai/gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "openai/gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    # Default fallback pricing
    "default": {
        "input": 0.10,
        "output": 0.40,
    }
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate estimated cost for LLM usage
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    # Get pricing for the model or use default
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    
    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


def format_cost_info(model_name: str, input_tokens: int, output_tokens: int, total_tokens: int) -> str:
    """
    Format cost information for display
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total number of tokens
        
    Returns:
        Formatted cost information string
    """
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    
    # Convert to Thai Baht (approximate rate: 1 USD = 33 THB)
    cost_thb = cost * 35
    
    return (
        f"Input tokens: {input_tokens:,}\n"
        f"Output tokens: {output_tokens:,}\n"
        f"Total tokens: {total_tokens:,}\n"
        f"Estimated cost: ${cost:.6f} (≈{cost_thb:.4f} บาท)"
    )