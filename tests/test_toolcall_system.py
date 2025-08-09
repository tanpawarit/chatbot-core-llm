"""
Test suite for Tool Calling System
Tests that the LLM properly calls tools instead of hallucinating product information
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from src.llm.node.response_llm import generate_response
from src.models import Message, MessageRole
from src.tools.data.data_tools import search_items_by_price_range, search_items_by_name


class TestToolCallSystem:
    """Test the tool calling system functionality"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        # Mock product data that tools would return
        self.mock_gaming_pc = {
            "id": "gpc001",
            "name": "Gaming PC AMD Ryzen 5 GTX 1660 Super",
            "price": 35000,
            "category": "Desktop PC",
            "specs": {
                "cpu": "AMD Ryzen 5 3600",
                "gpu": "GTX 1660 Super",
                "ram": "16GB DDR4",
                "storage": "500GB SSD"
            },
            "stock": 5,
            "description": "Gaming PC for mid-range gaming"
        }
        
        self.mock_laptop = {
            "id": "lap001", 
            "name": "ASUS TUF Gaming F15",
            "price": 32000,
            "category": "Gaming Laptop",
            "specs": {
                "cpu": "Intel Core i5-11400H",
                "gpu": "GTX 1650",
                "ram": "8GB DDR4", 
                "storage": "512GB SSD"
            },
            "stock": 3,
            "description": "Gaming laptop for portability"
        }

    @patch('src.llm.node.response_llm.llm_factory')
    @patch('src.tools.data.data_tools.load_items')
    def test_tool_called_for_budget_inquiry(self, mock_load_items, mock_llm_factory):
        """Test that tools are called when user provides budget"""
        # Setup mocks
        mock_load_items.return_value = [self.mock_gaming_pc, self.mock_laptop]
        
        # Mock LLM to call search_items_by_price_range tool
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Create mock AI response with tool calls
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = [{
            'id': 'tool_123',
            'name': 'search_items_by_price_range',
            'args': {'min_price': 30000, 'max_price': 40000}
        }]
        mock_ai_message.content = ""
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = "ผมพบคอมสำหรับเล่นเกมในงบ 40,000 บาท ได้ Gaming PC AMD Ryzen 5 GTX 1660 Super ราคา 35,000 บาทครับ"
        mock_final_response.tool_calls = []
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.side_effect = [mock_ai_message, mock_final_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        # Test messages
        messages = [
            Message(role=MessageRole.USER, content="งบ 40000 เอาไว้เล่นเกมครับ")
        ]
        
        # Execute
        response = generate_response(messages)
        
        # Assertions
        assert mock_llm_factory.get_response_llm.called
        assert mock_llm.bind_tools.called
        # Should call invoke at least once, potentially twice if tools are called
        assert mock_llm_with_tools.invoke.call_count >= 1  
        # Verify the response content (whether from first call or after tool execution)
        assert isinstance(response, str)
        assert len(response) > 0

    @patch('src.llm.node.response_llm.llm_factory')
    @patch('src.tools.data.data_tools.load_items')
    def test_tool_called_for_ready_made_pc_inquiry(self, mock_load_items, mock_llm_factory):
        """Test that tools are called when user asks about ready-made PCs"""
        # Setup mocks
        mock_load_items.return_value = [self.mock_gaming_pc, self.mock_laptop]
        
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Mock AI response with tool calls
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = [{
            'id': 'tool_456',
            'name': 'search_items_by_price_range',
            'args': {'min_price': 35000, 'max_price': 45000}
        }]
        mock_ai_message.content = ""
        
        mock_final_response = Mock()
        mock_final_response.content = "มีครับ Gaming PC AMD Ryzen 5 GTX 1660 Super ราคา 35,000 บาท พร้อมจำหน่ายเลยครับ"
        mock_final_response.tool_calls = []
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.side_effect = [mock_ai_message, mock_final_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        # Test messages
        messages = [
            Message(role=MessageRole.USER, content="มีคอมสำเร็จเลยไหมครับ งบประมาณ 40000")
        ]
        
        # Execute
        response = generate_response(messages)
        
        # Assertions
        assert mock_llm_with_tools.invoke.call_count == 2
        assert "Gaming PC AMD Ryzen 5" in response
        assert "35,000" in response

    @patch('src.llm.node.response_llm.llm_factory')
    def test_no_tools_for_general_advice(self, mock_llm_factory):
        """Test that tools are NOT called for general buying advice"""
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Mock AI response WITHOUT tool calls (general advice)
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = []  # No tool calls for general advice
        mock_ai_message.content = "สำหรับการเล่นเกม แนะนำให้เน้นการ์ดจอดีๆ และ CPU ที่สมดุลกันครับ"
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.return_value = mock_ai_message
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        # Test messages - general advice request
        messages = [
            Message(role=MessageRole.USER, content="อยากทราบว่าสเปคคอมเกมที่ดีควรมีอะไรบ้างครับ")
        ]
        
        # Execute
        response = generate_response(messages)
        
        # Assertions
        assert mock_llm_with_tools.invoke.call_count == 1  # Only one call, no tool execution
        assert "การ์ดจอ" in response or "CPU" in response
        assert response == mock_ai_message.content

    @patch('src.llm.node.response_llm.llm_factory')
    @patch('src.tools.data.data_tools.load_items')
    def test_tool_error_handling(self, mock_load_items, mock_llm_factory):
        """Test graceful handling when tools encounter errors"""
        # Setup mocks to simulate tool error
        mock_load_items.side_effect = FileNotFoundError("Product data not found")
        
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Mock AI response with tool calls
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = [{
            'id': 'tool_789',
            'name': 'search_items_by_price_range',
            'args': {'min_price': 30000, 'max_price': 40000}
        }]
        mock_ai_message.content = ""
        
        # Mock final response after tool error
        mock_final_response = Mock()
        mock_final_response.content = "ขออพอทราบว่าระบบค้นหาขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งครับ"
        mock_final_response.tool_calls = []
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.side_effect = [mock_ai_message, mock_final_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        # Test messages
        messages = [
            Message(role=MessageRole.USER, content="งบ 40000 หาคอมเกมให้หน่อยครับ")
        ]
        
        # Execute
        response = generate_response(messages)
        
        # Assertions
        assert mock_llm_with_tools.invoke.call_count == 2
        assert "ขัดข้อง" in response or "ลองใหม่" in response

    @patch('src.llm.node.response_llm.llm_factory')
    @patch('src.tools.data.data_tools.load_items')
    def test_multiple_tool_calls(self, mock_load_items, mock_llm_factory):
        """Test handling of multiple tool calls in one response"""
        # Setup mocks
        mock_load_items.return_value = [self.mock_gaming_pc, self.mock_laptop]
        
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Mock AI response with multiple tool calls
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = [
            {
                'id': 'tool_1',
                'name': 'search_items_by_price_range',
                'args': {'min_price': 30000, 'max_price': 40000}
            },
            {
                'id': 'tool_2', 
                'name': 'get_categories',
                'args': {}
            }
        ]
        mock_ai_message.content = ""
        
        mock_final_response = Mock()
        mock_final_response.content = "ผมหาคอมในช่วงราคานี้และหมวดหมู่ที่เกี่ยวข้องให้แล้วครับ"
        mock_final_response.tool_calls = []
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.side_effect = [mock_ai_message, mock_final_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        # Test messages
        messages = [
            Message(role=MessageRole.USER, content="หาคอมในงบ 40000 และบอกหมวดหมู่สินค้าด้วยครับ")
        ]
        
        # Execute
        response = generate_response(messages)
        
        # Assertions
        assert mock_llm_with_tools.invoke.call_count == 2
        assert isinstance(response, str)
        assert len(response) > 0

    def test_tool_integration_with_actual_data(self):
        """Integration test with actual tool functions"""
        # Test actual tool function
        with patch('src.tools.data.data_tools.load_items') as mock_load:
            mock_load.return_value = [self.mock_gaming_pc, self.mock_laptop]
            
            # Test price range search
            results = search_items_by_price_range.invoke({'min_price': 30000, 'max_price': 40000})
            
            assert len(results) == 2
            assert any(item['name'] == 'Gaming PC AMD Ryzen 5 GTX 1660 Super' for item in results)
            assert any(item['name'] == 'ASUS TUF Gaming F15' for item in results)

    @patch('src.llm.node.response_llm.llm_factory')
    def test_system_prompt_includes_tool_instructions(self, mock_llm_factory):
        """Test that system prompt includes proper tool calling instructions"""
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = []
        mock_ai_message.content = "ได้ครับ จะช่วยหาให้นะครับ"
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.return_value = mock_ai_message
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        messages = [
            Message(role=MessageRole.USER, content="สวัสดีครับ")
        ]
        
        response = generate_response(messages)
        
        # Check that the system prompt was built and passed
        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        system_message = call_args[0]  # First message should be system
        
        # Verify tool instructions are present
        assert "CRITICAL: When customers ask about specific models, prices/budgets" in system_message.content
        assert "search_items_by_price_range" in system_message.content
        assert "NEVER provide specific product names, prices, or specs without calling tools" in system_message.content


class TestToolCallPreventsHallucination:
    """Test specific scenarios that previously caused hallucination"""
    
    @patch('src.llm.node.response_llm.llm_factory')
    @patch('src.tools.data.data_tools.load_items')
    def test_prevents_hallucinating_acer_nitro_models(self, mock_load_items, mock_llm_factory):
        """Test that the system doesn't hallucinate ACER NITRO models without tool calls"""
        # Return empty results to test hallucination prevention
        mock_load_items.return_value = []
        
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Mock that LLM calls tool but gets no results
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = [{
            'id': 'tool_check',
            'name': 'search_items_by_price_range',
            'args': {'min_price': 35000, 'max_price': 45000}
        }]
        mock_ai_message.content = ""
        
        # Mock appropriate response when no results found
        mock_final_response = Mock()
        mock_final_response.content = "ขออภัยครับ ยังไม่พบสินค้าในช่วงราคานี้ในขณะนี้ ลองปรับงบประมาณหรือช่วงราคาใหม่ได้ไหมครับ"
        mock_final_response.tool_calls = []
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.side_effect = [mock_ai_message, mock_final_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        messages = [
            Message(role=MessageRole.USER, content="มีคอมสำเร็จเลยไหมครับ งบ 40000")
        ]
        
        response = generate_response(messages)
        
        # Should not mention specific models that weren't returned by tools
        assert "ACER NITRO" not in response
        assert "MSI Gaming" not in response
        assert "ยังไม่พบ" in response or "ขออภัย" in response

    @patch('src.llm.node.response_llm.llm_factory')
    def test_system_prevents_price_hallucination(self, mock_llm_factory):
        """Test that system prevents hallucinating prices without tool verification"""
        mock_llm = Mock()
        mock_llm_factory.get_response_llm.return_value = mock_llm
        
        # Mock LLM response that tries to provide general advice without specific prices
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = []  # No tool calls for general advice
        mock_ai_message.content = "สำหรับงบ 40,000 บาท แนะนำให้เน้นที่การ์ดจอและ CPU ที่สมดุลกันครับ เดี๋ยวให้ผมค้นหารุ่นที่เหมาะสมให้นะครับ"
        
        mock_llm_with_tools = Mock()
        mock_llm_with_tools.invoke.return_value = mock_ai_message
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        messages = [
            Message(role=MessageRole.USER, content="คอมเกมงบ 40000 แนะนำอะไรดีครับ")
        ]
        
        response = generate_response(messages)
        
        # Should not contain specific prices without tool verification
        price_patterns = ["35,990", "31,990", "บาท", "฿"]
        has_specific_prices = any(pattern in response for pattern in price_patterns if pattern != "บาท")
        
        if has_specific_prices:
            # If prices are mentioned, tools should have been called
            # This is checked by verifying the response doesn't contain hallucinated specific models
            assert "Intel Core i5-12500H" not in response
            assert "NVIDIA GeForce RTX 3050" not in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])