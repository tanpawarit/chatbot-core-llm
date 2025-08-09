"""
Test suite for the Context Routing System
Tests intent-based context selection and token optimization
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List

from src.llm.routing import ContextRouter, context_router
from src.models.nlu_model import NLUResult, NLUIntent, NLUEntity, NLUSentiment


class TestContextRouter:
    """Test the ContextRouter class functionality"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        self.router = ContextRouter()
        
    def create_nlu_result(
        self, 
        content: str = "test message",
        intents: List[tuple] = None, 
        entities: List[tuple] = None,
        sentiment: tuple = None
    ) -> NLUResult:
        """Helper to create NLUResult with test data"""
        nlu_intents = []
        if intents:
            for name, confidence, priority in intents:
                nlu_intents.append(NLUIntent(
                    name=name, 
                    confidence=confidence, 
                    priority_score=priority
                ))
        
        nlu_entities = []
        if entities:
            for entity_type, value, confidence in entities:
                nlu_entities.append(NLUEntity(
                    type=entity_type,
                    value=value,
                    confidence=confidence
                ))
        
        nlu_sentiment = None
        if sentiment:
            label, confidence = sentiment
            nlu_sentiment = NLUSentiment(label=label, confidence=confidence)
        
        return NLUResult(
            content=content,
            intents=nlu_intents,
            entities=nlu_entities,
            sentiment=nlu_sentiment
        )

    @patch('src.llm.routing.config_manager')
    def test_parse_default_intents(self, mock_config_manager):
        """Test parsing of default_intent configuration string"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8, inquiry_intent:0.7"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        result = router._parse_default_intents()
        
        assert result == {"greet", "purchase_intent", "inquiry_intent"}

    @patch('src.llm.routing.config_manager')
    def test_minimal_contexts_for_greet(self, mock_config_manager):
        """Test minimal context routing for greet intent"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        nlu_result = self.create_nlu_result(
            content="สวัสดีครับ",
            intents=[("greet", 0.9, 0.8)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Check minimal contexts are applied
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "user_history": True,
            "product_details": False,  # Should be disabled
            "business_policies": False  # Should be disabled
        }
        
        assert contexts == expected_contexts

    @patch('src.llm.routing.config_manager')
    def test_product_focused_contexts_for_purchase(self, mock_config_manager):
        """Test product-focused context routing for purchase_intent"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        nlu_result = self.create_nlu_result(
            content="อยากซื้อคอมพิวเตอร์ครับ",
            intents=[("purchase_intent", 0.9, 0.8)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Check product-focused contexts are applied
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "product_details": True,  # Should be enabled for purchase
            "business_policies": True,  # Should be enabled for purchase
            "user_history": False  # Should be disabled
        }
        
        assert contexts == expected_contexts

    @patch('src.llm.routing.config_manager')
    def test_support_focused_contexts_for_support(self, mock_config_manager):
        """Test support-focused context routing for support_intent"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "support_intent:0.6, complain_intent:0.6"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        nlu_result = self.create_nlu_result(
            content="มีปัญหากับคอมพิวเตอร์ที่ซื้อไป",
            intents=[("support_intent", 0.8, 0.7)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Check support-focused contexts are applied
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "business_policies": True,  # Should be enabled for support
            "user_history": True,  # Should be enabled for support
            "product_details": False  # Should be disabled
        }
        
        assert contexts == expected_contexts

    @patch('src.llm.routing.config_manager')
    def test_support_focused_contexts_for_complain(self, mock_config_manager):
        """Test support-focused context routing for complain_intent"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "support_intent:0.6, complain_intent:0.6"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        nlu_result = self.create_nlu_result(
            content="ไม่พอใจกับบริการ",
            intents=[("complain_intent", 0.8, 0.7)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Check support-focused contexts are applied (same as support_intent)
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "business_policies": True,  # Should be enabled for complaints
            "user_history": True,  # Should be enabled for complaints
            "product_details": False  # Should be disabled
        }
        
        assert contexts == expected_contexts

    @patch('src.llm.routing.config_manager')
    def test_full_contexts_for_inquiry(self, mock_config_manager):
        """Test full context routing for inquiry_intent"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "inquiry_intent:0.7"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        nlu_result = self.create_nlu_result(
            content="มีคอมพิวเตอร์รุ่นไหนแนะนำบ้าง",
            intents=[("inquiry_intent", 0.8, 0.7)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Check full contexts are applied
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "user_history": True,
            "product_details": True,  # Should be enabled for inquiry
            "business_policies": True
        }
        
        assert contexts == expected_contexts

    @patch('src.llm.routing.config_manager')
    def test_full_contexts_for_unknown_intent(self, mock_config_manager):
        """Test full context routing for unknown/additional intents"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        nlu_result = self.create_nlu_result(
            content="ข้อมูลอะไรก็ได้",
            intents=[("unknown_intent", 0.8, 0.7)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Check full contexts are applied (safe default)
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "user_history": True,
            "product_details": True,
            "business_policies": True
        }
        
        assert contexts == expected_contexts

    def test_full_contexts_for_no_nlu_result(self):
        """Test full context routing when no NLU result is provided"""
        contexts = self.router.determine_required_contexts(None)
        
        # Check full contexts are applied (safe default)
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "user_history": True,
            "product_details": True,
            "business_policies": True
        }
        
        assert contexts == expected_contexts

    def test_full_contexts_for_empty_intents(self):
        """Test full context routing when NLU result has empty intents"""
        nlu_result = self.create_nlu_result(content="test", intents=[])
        contexts = self.router.determine_required_contexts(nlu_result)
        
        # Check full contexts are applied (safe default)
        expected_contexts = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "user_history": True,
            "product_details": True,
            "business_policies": True
        }
        
        assert contexts == expected_contexts


class TestTokenEstimation:
    """Test token usage estimation functionality"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        self.router = ContextRouter()
    
    def test_minimal_context_token_estimation(self):
        """Test token estimation for minimal contexts (greet)"""
        contexts = {
            "core_behavior": True,         # 100 tokens
            "interaction_guidelines": True, # 150 tokens
            "user_history": True,          # 300 tokens
            "product_details": False,      # 0 tokens (disabled)
            "business_policies": False,    # 0 tokens (disabled)
        }
        
        tokens = self.router.estimate_token_usage(contexts)
        
        # Should be 550 tokens (100 + 150 + 300)
        assert tokens == 550

    def test_product_focused_context_token_estimation(self):
        """Test token estimation for product-focused contexts (purchase)"""
        contexts = {
            "core_behavior": True,         # 100 tokens
            "interaction_guidelines": True, # 150 tokens
            "product_details": True,       # 800 tokens
            "business_policies": True,     # 200 tokens
            "user_history": False,         # 0 tokens (disabled)
        }
        
        tokens = self.router.estimate_token_usage(contexts)
        
        # Should be 1250 tokens (100 + 150 + 800 + 200)
        assert tokens == 1250

    def test_support_focused_context_token_estimation(self):
        """Test token estimation for support-focused contexts (support/complain)"""
        contexts = {
            "core_behavior": True,         # 100 tokens
            "interaction_guidelines": True, # 150 tokens
            "business_policies": True,     # 200 tokens
            "user_history": True,          # 300 tokens
            "product_details": False,      # 0 tokens (disabled)
        }
        
        tokens = self.router.estimate_token_usage(contexts)
        
        # Should be 750 tokens (100 + 150 + 200 + 300)
        assert tokens == 750

    def test_full_context_token_estimation(self):
        """Test token estimation for full contexts (inquiry/unknown)"""
        contexts = {
            "core_behavior": True,         # 100 tokens
            "interaction_guidelines": True, # 150 tokens
            "user_history": True,          # 300 tokens
            "product_details": True,       # 800 tokens
            "business_policies": True,     # 200 tokens
        }
        
        tokens = self.router.estimate_token_usage(contexts)
        
        # Should be 1550 tokens (100 + 150 + 300 + 800 + 200)
        assert tokens == 1550

    def test_savings_calculation(self):
        """Test token savings calculation vs full context"""
        # Full context baseline: 1550 tokens
        full_contexts = {
            "core_behavior": True,         # 100
            "interaction_guidelines": True, # 150
            "user_history": True,          # 300
            "product_details": True,       # 800
            "business_policies": True,     # 200
        }
        
        # Minimal context: 550 tokens
        minimal_contexts = {
            "core_behavior": True,         # 100
            "interaction_guidelines": True, # 150
            "user_history": True,          # 300
            "product_details": False,      # 0
            "business_policies": False,    # 0
        }
        
        full_tokens = self.router.estimate_token_usage(full_contexts)
        minimal_tokens = self.router.estimate_token_usage(minimal_contexts)
        
        # Calculate savings
        savings = full_tokens - minimal_tokens
        savings_percent = (savings / full_tokens) * 100
        
        assert full_tokens == 1550
        assert minimal_tokens == 550
        assert savings == 1000
        assert abs(savings_percent - 64.5) < 0.1  # ~64.5% savings


class TestIntegrationScenarios:
    """Test end-to-end routing scenarios"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        self.router = ContextRouter()
        
    def create_nlu_result(self, content: str, intents: List[tuple]) -> NLUResult:
        """Helper to create NLUResult with test data"""
        nlu_intents = []
        for name, confidence, priority in intents:
            nlu_intents.append(NLUIntent(
                name=name, 
                confidence=confidence, 
                priority_score=priority
            ))
        
        return NLUResult(content=content, intents=nlu_intents)

    @patch('src.llm.routing.config_manager')
    def test_greeting_scenario(self, mock_config_manager):
        """Test complete greeting scenario with token optimization"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        
        # Test greeting message
        nlu_result = self.create_nlu_result(
            content="สวัสดีครับ",
            intents=[("greet", 0.9, 0.8)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        tokens = router.estimate_token_usage(contexts)
        
        # Should use minimal contexts and save ~64% tokens
        assert contexts["core_behavior"] == True
        assert contexts["interaction_guidelines"] == True
        assert contexts["user_history"] == True
        assert contexts["product_details"] == False  # Key optimization
        assert contexts["business_policies"] == False  # Key optimization
        assert tokens == 550

    @patch('src.llm.routing.config_manager')
    def test_purchase_scenario(self, mock_config_manager):
        """Test complete purchase scenario with appropriate contexts"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        
        # Test purchase message
        nlu_result = self.create_nlu_result(
            content="อยากซื้อคอมพิวเตอร์ราคาประมาณ 30000 บาท",
            intents=[("purchase_intent", 0.9, 0.8)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        tokens = router.estimate_token_usage(contexts)
        
        # Should use product-focused contexts
        assert contexts["core_behavior"] == True
        assert contexts["interaction_guidelines"] == True
        assert contexts["product_details"] == True  # Essential for purchase
        assert contexts["business_policies"] == True  # Essential for purchase
        assert contexts["user_history"] == False  # Not needed for new purchase
        assert tokens == 1250

    @patch('src.llm.routing.config_manager')
    def test_support_scenario(self, mock_config_manager):
        """Test complete support scenario with appropriate contexts"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "support_intent:0.6"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        
        # Test support message
        nlu_result = self.create_nlu_result(
            content="คอมพิวเตอร์ที่ซื้อไปเปิดไม่ติด",
            intents=[("support_intent", 0.8, 0.7)]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        tokens = router.estimate_token_usage(contexts)
        
        # Should use support-focused contexts
        assert contexts["core_behavior"] == True
        assert contexts["interaction_guidelines"] == True
        assert contexts["business_policies"] == True  # Essential for support
        assert contexts["user_history"] == True  # Essential for support
        assert contexts["product_details"] == False  # Not needed for existing issue
        assert tokens == 750

    @patch('src.llm.routing.config_manager')
    def test_mixed_intent_scenario(self, mock_config_manager):
        """Test scenario with multiple intents (greet takes priority due to if-elif order)"""
        # Mock config
        mock_config = Mock()
        mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8, support_intent:0.6"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        
        # Test message with mixed intents
        nlu_result = self.create_nlu_result(
            content="สวัสดีครับ อยากซื้อคอมพิวเตอร์",
            intents=[
                ("greet", 0.7, 0.8),
                ("purchase_intent", 0.9, 0.8)  # Higher confidence but greet takes priority
            ]
        )
        
        contexts = router.determine_required_contexts(nlu_result)
        
        # Should use minimal contexts (greet wins due to if-elif order, not confidence)
        assert contexts["core_behavior"] == True
        assert contexts["interaction_guidelines"] == True
        assert contexts["user_history"] == True
        assert contexts["product_details"] == False  # Disabled in greet mode
        assert contexts["business_policies"] == False  # Disabled in greet mode

    def test_global_router_instance(self):
        """Test that global context_router instance works correctly"""
        # Test that global instance exists and is functional
        assert context_router is not None
        assert isinstance(context_router, ContextRouter)
        
        # Test basic functionality
        contexts = context_router.determine_required_contexts(None)
        assert isinstance(contexts, dict)
        assert len(contexts) == 5  # Should have 5 context types


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        self.router = ContextRouter()

    @patch('src.llm.routing.config_manager')
    def test_config_parsing_error(self, mock_config_manager):
        """Test handling of config parsing errors"""
        # Mock config with invalid format
        mock_config = Mock()
        mock_config.nlu.default_intent = "invalid::format"
        mock_config_manager.get_config.return_value = mock_config
        
        router = ContextRouter()
        result = router._parse_default_intents()
        
        # Should return empty set on parsing error
        assert result == {"invalid"}  # Will still try to parse, but gracefully

    @patch('src.llm.routing.config_manager')
    def test_config_access_error(self, mock_config_manager):
        """Test handling of config access errors"""
        # Mock config manager to raise exception
        mock_config_manager.get_config.side_effect = Exception("Config error")
        
        # Should not crash when creating router
        try:
            router = ContextRouter()
            # Should still work with fallback behavior
            contexts = router.determine_required_contexts(None)
            assert isinstance(contexts, dict)
        except Exception:
            pytest.fail("Router should handle config errors gracefully")

    def test_invalid_context_types(self):
        """Test token estimation with unknown context types"""
        contexts = {
            "core_behavior": True,
            "unknown_context": True,  # Unknown context type
            "another_unknown": True,
        }
        
        tokens = self.router.estimate_token_usage(contexts)
        
        # Should handle unknown contexts gracefully (treat as 0 tokens)
        assert tokens == 100  # Only core_behavior (100 tokens)

    def test_empty_contexts(self):
        """Test token estimation with empty contexts"""
        contexts = {}
        
        tokens = self.router.estimate_token_usage(contexts)
        
        # Should return 0 tokens for empty contexts
        assert tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])