# 🚨 API Speed Issues & Solutions

## 📊 **ปัญหาที่พบ**
- Warmup timeout หลัง 30 วินาที  
- LLM API response ช้ามาก (>30s)
- Cold start กินเวลานานเกินไป

## 🔍 **สาเหตุที่เป็นไปได้**

### 1. Network/API Issues
- OpenRouter API ช้า
- Network latency สูง
- Rate limiting จาก API provider

### 2. Model Issues  
- Models ใหญ่เกินไป:
  - `mistralai/mistral-small-3.2-24b-instruct` (24B parameters!)
  - `google/gemini-2.5-flash-lite`

### 3. Configuration Issues
- Request timeout ไม่เหมาะสม
- HTTP client settings

## 🛠️ **Solutions**

### 🚀 **Solution 1: เปลี่ยน Models เร็วขึ้น**

แก้ไขใน `config.yaml`:
```yaml
openrouter:
  classification:
    model: "google/gemini-flash-1.5"  # เร็วกว่า gemini-2.5
    temperature: 0.1
  response:  
    model: "google/gemini-flash-1.5"  # เร็วกว่า mistral-small-3.2-24b
    temperature: 0.7
```

### 🔧 **Solution 2: ปรับ HTTP Settings**

เพิ่มใน `factory.py`:
```python
import httpx

# Custom HTTP client with faster settings
http_client = httpx.Client(
    timeout=httpx.Timeout(30.0),  # Explicit timeout
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0
    )
)

self._instances[key] = ChatOpenAI(
    model=config.response.model,
    api_key=convert_to_secret_str(config.api_key),
    base_url=config.base_url,
    temperature=config.response.temperature,
    http_client=http_client,  # Use custom client
    request_timeout=30,       # 30s timeout
    max_retries=1,           # Reduce retries
)
```

### ⚡ **Solution 3: Optional Warmup**

เพิ่มใน `main.py`:
```python
# Optional warmup with user choice
print("🔥 LLM Warmup available (may take 30+ seconds)")
choice = input("Skip warmup for faster start? (y/N): ").strip().lower()

if choice != 'y':
    print("🔥 Warming up LLMs...")
    llm_factory.warmup_all_llms()
else:
    print("⚡ Skipping warmup - first response may be slower")
```

### 🏃‍♂️ **Solution 4: Background Warmup**

```python
import threading

def background_warmup():
    """Run warmup in background"""
    threading.Thread(target=llm_factory.warmup_all_llms, daemon=True).start()

# In main.py
print("🔥 Starting background warmup...")
background_warmup()
# Continue immediately without waiting
```

## 🎯 **Recommended Action Plan**

### Phase 1: Quick Fixes (5 mins)
1. ✅ เปลี่ยนเป็น `gemini-flash-1.5` (เร็วกว่า)
2. ✅ เพิ่ม optional warmup choice
3. ✅ ลด warmup timeout เหลือ 15s

### Phase 2: Advanced Fixes (15 mins)  
1. Custom HTTP client settings
2. Background warmup
3. Connection pooling

### Phase 3: Alternative Approaches (30 mins)
1. Local model caching
2. Response caching system  
3. Streaming responses

## 🧪 **Testing Commands**

```bash
# Test different approaches
python quick_warmup_test.py           # Current approach
python main.py                        # With warmup
python skip_warmup_main.py            # Without warmup
```

## 📈 **Expected Results**

With faster models:
- Warmup: 5-15 seconds (vs 30+ seconds)
- First response: 3-8 seconds (vs 30+ seconds) 
- Subsequent responses: 1-3 seconds

ให้เริ่มจาก Phase 1 ก่อน!