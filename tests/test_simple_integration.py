"""
Simple Integration Tests for the System
Tests core functionality without complex mocking
"""

import pytest
from unittest.mock import patch
from src.llm.node.nlu_llm import analyze_message_nlu
from src.models import NLUResult
from src.config import config_manager


class TestSimpleIntegration:
    """Simple integration tests for core system functionality"""
    
    def test_importance_scoring_works(self):
        """Test that importance scoring now works for business messages"""
        # Test various business messages
        test_cases = [
            ("เอาครบชุดเลยครับ", True),
            ("งบ 40000 เอาไว้เล่นเกมครับ", True), 
            ("มีคอมสำเร็จเลยไหมครับ", True),
            ("อยากจัดสเปคคอมครับ", True),
            ("สวัสดีครับ", True),  # Should now pass threshold with new algorithm
        ]
        
        threshold = config_manager.get_nlu_config().importance_threshold
        
        for message, should_pass in test_cases:
            # Use actual NLU analysis
            nlu_result = analyze_message_nlu(message)
            
            # Check importance score
            passes_threshold = nlu_result.importance_score >= threshold
            
            print(f"Message: '{message}' → Score: {nlu_result.importance_score:.3f} (threshold: {threshold}) → {'✅' if passes_threshold else '❌'}")
            
            if should_pass:
                assert passes_threshold, f"Message '{message}' should pass threshold but got score {nlu_result.importance_score}"
            
            # Verify it's a proper NLUResult
            assert isinstance(nlu_result, NLUResult)
            assert nlu_result.content == message
            assert nlu_result.importance_score >= 0.0
            assert nlu_result.importance_score <= 1.0

    def test_nlu_analysis_structure(self):
        """Test that NLU analysis returns proper structure"""
        message = "งบ 40000 เอาไว้เล่นเกมครับ"
        nlu_result = analyze_message_nlu(message)
        
        # Check structure
        assert hasattr(nlu_result, 'content')
        assert hasattr(nlu_result, 'intents') 
        assert hasattr(nlu_result, 'entities')
        assert hasattr(nlu_result, 'importance_score')
        assert hasattr(nlu_result, 'primary_intent')
        
        # Check content
        assert nlu_result.content == message
        assert isinstance(nlu_result.intents, list)
        assert isinstance(nlu_result.entities, list) 
        assert isinstance(nlu_result.importance_score, float)
        
        # Check primary intent extraction works
        if nlu_result.intents:
            assert isinstance(nlu_result.primary_intent, str)
        else:
            assert nlu_result.primary_intent is None

    @patch('src.tools.data.data_tools.load_items')
    def test_tool_integration_basic(self, mock_load_items):
        """Test that tool functions can be called successfully"""
        # Mock product data
        mock_products = [
            {
                "id": "test1",
                "name": "Test Gaming PC",
                "price": 35000,
                "category": "Desktop PC",
                "stock": 5
            }
        ]
        mock_load_items.return_value = mock_products
        
        # Test tool import and basic functionality
        from src.tools.data.data_tools import search_items_by_price_range
        
        # Test the tool function directly using invoke method
        results = search_items_by_price_range.invoke({'min_price': 30000, 'max_price': 40000})
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]['name'] == 'Test Gaming PC'
        assert results[0]['price'] == 35000

    def test_config_loading(self):
        """Test that configuration loads properly"""
        config = config_manager.get_config()
        
        # Check basic config structure
        assert hasattr(config, 'nlu')
        assert hasattr(config, 'openrouter')
        assert hasattr(config, 'memory')
        
        # Check NLU config
        nlu_config = config_manager.get_nlu_config()
        assert hasattr(nlu_config, 'importance_threshold')
        assert hasattr(nlu_config, 'default_intent')
        assert hasattr(nlu_config, 'additional_intent')
        
        # Check that threshold is the new value
        assert nlu_config.importance_threshold == 0.35

    def test_business_keyword_detection(self):
        """Test that business keywords are properly detected"""
        from src.models.nlu_model import NLUResult, NLUIntent
        
        # Test with business keywords
        business_message = "งบ 40000 บาท อยากได้คอมครับ"
        nlu_result = NLUResult(
            content=business_message,
            intents=[NLUIntent(name='purchase_intent', confidence=0.8, priority_score=0.8)]
        )
        
        # Should get high score due to business keywords: งบ, บาท, อยาก, ครับ
        score = nlu_result.importance_score
        print(f"Business message score: {score:.3f}")
        
        assert score > 0.5, f"Business message should get high score, got {score}"
        
        # Test with non-business message
        generic_message = "สวัสดี"
        nlu_result2 = NLUResult(
            content=generic_message,
            intents=[NLUIntent(name='greet', confidence=0.9, priority_score=0.3)]
        )
        
        score2 = nlu_result2.importance_score
        print(f"Generic message score: {score2:.3f}")
        
        # Both should pass threshold, but business message should score higher
        assert score > score2, f"Business message ({score}) should score higher than generic ({score2})"

    def test_system_end_to_end_basic(self):
        """Basic end-to-end test without complex mocking"""
        # Test the main workflow components work together
        message = "อยากจัดสเปคคอมครับ"
        
        # Step 1: NLU Analysis
        nlu_result = analyze_message_nlu(message)
        assert isinstance(nlu_result, NLUResult)
        
        # Step 2: Check importance scoring
        threshold = config_manager.get_nlu_config().importance_threshold
        assert nlu_result.importance_score >= threshold
        
        # Step 3: Check that we get business intent
        assert nlu_result.primary_intent in ['purchase_intent', 'inquiry_intent', 'general_intent']
        
        print(f"✅ End-to-end test passed:")
        print(f"   Message: {message}")
        print(f"   Intent: {nlu_result.primary_intent}")
        print(f"   Score: {nlu_result.importance_score:.3f} (threshold: {threshold})")


class TestSystemHealth:
    """Health checks for the system"""
    
    def test_all_imports_work(self):
        """Test that all main modules can be imported"""
        # Core modules
        from src.models import Message, NLUResult, Conversation
        from src.llm.node.nlu_llm import analyze_message_nlu
        from src.llm.node.response_llm import generate_response
        from src.config import config_manager
        from src.memory.manager import memory_manager
        
        # Tools
        from src.tools.data.data_tools import search_items_by_price_range
        
        assert True  # If we get here, imports worked

    def test_environment_setup(self):
        """Test that environment is set up correctly"""
        import os
        
        # Check for .env file presence (not content, just existence)
        env_path = os.path.join(os.getcwd(), '.env')
        if not os.path.exists(env_path):
            pytest.skip("No .env file found, skipping environment test")
        
        # Test config can be loaded
        config = config_manager.get_config()
        assert config is not None

    def test_data_directory_exists(self):
        """Test that required data directories exist"""
        import os
        
        required_paths = [
            'data',
            'data/longterm', 
            'data/product_detail'
        ]
        
        for path in required_paths:
            full_path = os.path.join(os.getcwd(), path)
            assert os.path.exists(full_path), f"Required directory {path} not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])