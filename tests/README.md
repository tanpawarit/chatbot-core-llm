# Test Suite Documentation

This directory contains comprehensive tests for the POC Chat System components.

## Overview

The test suite validates the core functionality of the dual-memory chatbot system, with particular focus on the intelligent context routing system that provides 19-64% token optimization.

## Test Structure

```
tests/
├── README.md                    # This file
├── test_routing_system.py      # Context routing system tests (23 tests)
└── run_tests.py                # Custom test runner script
```

## Test Coverage

### 🎯 Context Routing System (`test_routing_system.py`)

**Core Components Tested:**
- **Intent-based Context Selection**: Validates proper context routing based on NLU intents
- **Token Optimization**: Verifies 19-64% token savings across different intent types
- **Configuration Management**: Tests config parsing and error handling
- **Integration Scenarios**: End-to-end workflow validation

**Test Classes:**

#### `TestContextRouter` (9 tests)
Tests the core routing logic and configuration handling.

```python
# Key validations:
- Intent parsing from config.yaml
- Context selection for each intent type:
  • greet → minimal contexts (550 tokens, 64% savings)
  • purchase_intent → product-focused (1,250 tokens, 19% savings)
  • support_intent/complain_intent → support-focused (750 tokens, 52% savings)
  • inquiry_intent → full contexts (1,550 tokens)
  • unknown_intent → full contexts (safe fallback)
```

#### `TestTokenEstimation` (5 tests)
Validates token counting accuracy and savings calculations.

```python
# Token estimates per context type:
- core_behavior: 100 tokens
- interaction_guidelines: 150 tokens
- product_details: 800 tokens (most expensive)
- business_policies: 200 tokens
- user_history: 300 tokens
```

#### `TestIntegrationScenarios` (5 tests)
End-to-end testing of realistic chatbot scenarios.

```python
# Scenarios tested:
- Thai greeting: "สวัสดีครับ" → minimal contexts
- Purchase inquiry: "อยากซื้อคอมพิวเตอร์" → product-focused
- Support request: "คอมพิวเตอร์เปิดไม่ติด" → support-focused
- Mixed intents: Priority handling validation
```

#### `TestEdgeCases` (4 tests)
Error handling and edge case validation.

```python
# Edge cases covered:
- Configuration parsing errors
- Missing configuration fallback
- Invalid context types
- Empty input handling
```

## Running Tests

### Method 1: Custom Test Runner (Recommended)
```bash
# Run all routing system tests
python tests/run_tests.py

# Run specific test class
python tests/run_tests.py TestContextRouter
python tests/run_tests.py TestTokenEstimation
python tests/run_tests.py TestIntegrationScenarios
python tests/run_tests.py TestEdgeCases
```

### Method 2: Direct pytest
```bash
# All routing tests
uv run python -m pytest tests/test_routing_system.py -v

# Specific test method
uv run python -m pytest tests/test_routing_system.py::TestContextRouter::test_minimal_contexts_for_greet -v

# With coverage (if coverage tools installed)
uv run python -m pytest tests/test_routing_system.py --cov=src.llm.routing
```

## Test Results Interpretation

### ✅ Success Indicators
- **23/23 tests passing**: All routing logic working correctly
- **Token savings validated**: 19-64% optimization confirmed
- **Error handling robust**: Graceful degradation on failures
- **Integration working**: End-to-end scenarios successful

### ❌ Failure Investigation
If tests fail, check:

1. **Configuration Issues**: 
   - Ensure `config.yaml` contains valid `nlu.default_intent` format
   - Check environment variables are properly set

2. **Import Errors**:
   - Verify all dependencies installed: `uv sync`
   - Check Python path includes project root

3. **Logic Changes**:
   - If routing logic modified, update corresponding tests
   - Ensure token estimates match actual implementation

## Test Data

### Mock NLU Results
Tests use realistic Thai language examples:

```python
# Greeting scenarios
"สวัสดีครับ" → greet intent
"หวัดดี" → greet intent

# Purchase scenarios  
"อยากซื้อคอมพิวเตอร์ครับ" → purchase_intent
"ราคาเท่าไหร่" → purchase_intent

# Support scenarios
"มีปัญหากับคอมพิวเตอร์" → support_intent
"ไม่พอใจบริการ" → complain_intent
```

### Expected Token Usage
Based on context combinations:

| Intent Type | Contexts Enabled | Token Count | Savings |
|-------------|------------------|-------------|---------|
| greet | core + interaction + history | 550 | 64.5% |
| purchase | core + interaction + product + policies | 1,250 | 19.4% |
| support | core + interaction + policies + history | 750 | 51.6% |
| inquiry | all contexts | 1,550 | 0% |
| unknown | all contexts (safe) | 1,550 | 0% |

## Adding New Tests

### For New Intent Types
1. Add test cases in `TestContextRouter`
2. Update token estimation tests if new contexts added
3. Add integration scenario test
4. Update this README

### For New Features
1. Create new test class following existing patterns
2. Use descriptive test names: `test_feature_scenario_expected_result`
3. Include both positive and negative test cases
4. Add edge case testing

### Test Writing Guidelines

```python
def test_descriptive_name(self):
    """Clear description of what this test validates"""
    # Arrange
    setup_test_data()
    
    # Act
    result = system_under_test()
    
    # Assert
    assert result == expected_value
    assert side_effects_occurred()
```

## Integration with CI/CD

The test suite is designed for integration with continuous integration:

```yaml
# Example GitHub Actions step
- name: Run Routing System Tests
  run: |
    uv sync
    python tests/run_tests.py
```

## Dependencies

### Required for Testing
- `pytest>=8.0.0` - Test framework
- `unittest.mock` - Mocking capabilities (built-in)

### Production Dependencies Tested
- `src.llm.routing` - Context routing system
- `src.models.nlu_model` - NLU data models
- `src.config.manager` - Configuration management

## Performance Considerations

### Test Execution Time
- **Full suite**: ~0.1 seconds (23 tests)
- **Individual test**: ~5ms average
- **Memory usage**: Minimal (mock objects only)

### Optimization Notes
- Tests use mocked config to avoid file I/O
- NLU results created programmatically (no API calls)
- Parallel execution safe (no shared state)

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Solution: Ensure project root in Python path
export PYTHONPATH=/Users/pawarison/dev/poc_chat_chat:$PYTHONPATH
```

**Config Mock Issues:**
```python
# Ensure mock structure matches real config
mock_config.nlu.default_intent = "greet:0.3, purchase_intent:0.8"
```

**Test Isolation:**
- Each test uses fresh router instance
- Mock decorators ensure no cross-test contamination
- Setup methods reset state

### Debug Mode
```bash
# Verbose output with debugging
uv run python -m pytest tests/test_routing_system.py -v -s --tb=long
```

## Contributing

When adding tests:

1. **Follow Existing Patterns**: Use similar structure and naming
2. **Test Edge Cases**: Include error conditions and boundary values  
3. **Update Documentation**: Add new tests to this README
4. **Validate Coverage**: Ensure new features have corresponding tests
5. **Thai Language Support**: Include realistic Thai examples in test data

## Future Enhancements

Planned test expansions:

- [ ] Memory system integration tests
- [ ] LLM processor pipeline tests  
- [ ] End-to-end conversation flow tests
- [ ] Performance benchmarking tests
- [ ] Multi-language routing tests
- [ ] Real NLU API integration tests (optional)

---

*Last Updated: 2025-08-07*  
*Test Suite Version: 1.0*  
*Coverage: Context Routing System (Complete)*