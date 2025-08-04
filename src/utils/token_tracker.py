"""
Token Usage Tracking Utility
Centralized token usage tracking and cost calculation
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import BaseMessage
from src.utils.cost_calculator import format_cost_info
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage data structure"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    timestamp: datetime
    operation_type: str  # 'classification', 'response', 'custom'
    cost_info: Optional[str] = None


class TokenTracker:
    """
    Centralized token usage tracking and reporting.
    Eliminates duplicate token tracking code across LLM nodes.
    """
    
    def __init__(self):
        self.usage_history: list[TokenUsage] = []
    
    def track_response(self, response: BaseMessage, model: str, operation_type: str = "unknown") -> Optional[TokenUsage]:
        """
        Track token usage from LLM response and calculate costs
        
        Args:
            response: BaseMessage response from LLM
            model: Model name used
            operation_type: Type of operation ('classification', 'response', etc.)
            
        Returns:
            TokenUsage object if successful, None otherwise
        """
        try:
            if not hasattr(response, 'usage_metadata') or not response.usage_metadata:
                logger.warning("No usage metadata available", model=model, operation_type=operation_type)
                return None
            
            usage = response.usage_metadata
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
            
            if not (input_tokens or output_tokens):
                logger.warning("No token usage data available", model=model, operation_type=operation_type)
                return None
            
            # Calculate cost information
            cost_info = format_cost_info(model, input_tokens, output_tokens, total_tokens)
            
            # Create usage record
            usage_record = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=model,
                timestamp=datetime.utcnow(),
                operation_type=operation_type,
                cost_info=cost_info
            )
            
            # Store in history
            self.usage_history.append(usage_record)
            
            logger.info("Token usage tracked", 
                       model=model,
                       operation_type=operation_type,
                       input_tokens=input_tokens,
                       output_tokens=output_tokens,
                       total_tokens=total_tokens)
            
            return usage_record
            
        except Exception as e:
            logger.error("Failed to track token usage", 
                        model=model, 
                        operation_type=operation_type,
                        error=str(e))
            return None
    
    def print_usage(self, usage: TokenUsage, emoji: str = "💰") -> None:
        """
        Print formatted token usage information
        
        Args:
            usage: TokenUsage object to print
            emoji: Emoji to use in output
        """
        print(f"{emoji} {usage.operation_type.title()} Usage:")
        if usage.cost_info:
            print(usage.cost_info)
        else:
            print(f"   Model: {usage.model}")
            print(f"   Input: {usage.input_tokens:,} tokens")
            print(f"   Output: {usage.output_tokens:,} tokens")
            print(f"   Total: {usage.total_tokens:,} tokens")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session-wide token usage statistics
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.usage_history:
            return {"total_usage": 0, "operations": 0, "models_used": [], "by_operation": {}}
        
        total_tokens = sum(usage.total_tokens for usage in self.usage_history)
        total_input = sum(usage.input_tokens for usage in self.usage_history)
        total_output = sum(usage.output_tokens for usage in self.usage_history)
        
        models_used = list(set(usage.model for usage in self.usage_history))
        
        by_operation = {}
        for usage in self.usage_history:
            op_type = usage.operation_type
            if op_type not in by_operation:
                by_operation[op_type] = {
                    "count": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            
            by_operation[op_type]["count"] += 1
            by_operation[op_type]["total_tokens"] += usage.total_tokens
            by_operation[op_type]["input_tokens"] += usage.input_tokens
            by_operation[op_type]["output_tokens"] += usage.output_tokens
        
        return {
            "total_tokens": total_tokens,
            "total_input": total_input,
            "total_output": total_output,
            "operations": len(self.usage_history),
            "models_used": models_used,
            "by_operation": by_operation,
            "session_start": self.usage_history[0].timestamp if self.usage_history else None,
            "session_end": self.usage_history[-1].timestamp if self.usage_history else None
        }
    
    def print_session_summary(self) -> None:
        """Print session usage summary"""
        stats = self.get_session_stats()
        
        if stats["operations"] == 0:
            print("📊 No token usage recorded this session")
            return
        
        print("\n" + "="*50)
        print("📊 Session Token Usage Summary")
        print("="*50)
        print(f"Total Operations: {stats['operations']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Input Tokens: {stats['total_input']:,}")
        print(f"Output Tokens: {stats['total_output']:,}")
        print(f"Models Used: {', '.join(stats['models_used'])}")
        
        if stats["by_operation"]:
            print("\nBy Operation Type:")
            for op_type, op_stats in stats["by_operation"].items():
                print(f"  {op_type.title()}: {op_stats['count']} ops, {op_stats['total_tokens']:,} tokens")
        
        print("="*50)
    
    def clear_history(self) -> None:
        """Clear usage history"""
        logger.info("Clearing token usage history", records=len(self.usage_history))
        self.usage_history.clear()


# Global token tracker instance  
token_tracker = TokenTracker()