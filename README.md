  
```mermaid
flowchart TD
      Start([User Message]) --> CheckSM{SM Valid?}

      CheckSM -->|Yes| LoadSM[SM Loaded]
      CheckSM -->|No| LoadLM[Load LM from JSON]

      LoadSM --> AddMessage[Add Message to SM]

      LoadLM --> CreateSM{LM Found?}
      CreateSM -->|Yes| NewSM[SM Created from LM]
      CreateSM -->|No| NewConv[New Conversation]

      NewSM --> SaveSM[Save SM to Redis]
      NewConv --> SaveSM

      SaveSM --> AddMessage

      AddMessage --> ClassifyEvent[/LLM Classification/]

      ClassifyEvent --> CheckImportance{Important ≥0.7?}

      CheckImportance -->|Yes| SaveLM[Save to LM]
      CheckImportance -->|No| SkipLM[Skip LM Save]

      SaveLM --> GenerateResponse[/LLM Response Generation/]
      SkipLM --> GenerateResponse

      GenerateResponse --> AddResponse[Add Response to SM]
      AddResponse --> Complete([Complete])
```

============================================================
🤖 Classification LLM Context
============================================================
1. [SYSTEM] <system_identity>
Conversation analysis expert specializing in event classification and importance assessment.
</system_identity>

<event_types>
• INQUIRY: Questions, inquiries, asking for information
• FEEDBACK: Reviews, opinions, likes/dislikes, evaluations
• REQUEST: Requests for services, bookings, wanting something
• COMPLAINT: Problems, issues, complaints, dissatisfaction
• TRANSACTION: Buying, paying, pricing, financial matters
• SUPPORT: Help requests, guidance, how-to questions
• INFORMATION: Providing information, announcements, notifications
• GENERIC_EVENT: Greetings, thanks, social interactions
</event_types>

<importance_scale>
• 0.9-1.0: Transactions, critical issues
• 0.7-0.8: Important requests, feedback
• 0.5-0.6: Support requests
• 0.3-0.4: Simple questions
• 0.1-0.2: Greetings, social interactions
</importance_scale>

<output_format>
Respond with JSON following EventClassification schema only:
{
  "event_type": "one of the types above",
  "importance_score": 0.0-1.0,
  "intent": "text description of user intent",
  "reasoning": "brief explanation"
}
</output_format>
2. [HUMAN] สวัสดี
============================================================

============================================================
🤖 Response LLM Context
============================================================
1. [SYSTEM] <instructions>
คุณเป็นผู้ช่วยขายสินค้าคอมพิวเตอร์ที่เป็นมิตรและมีความรู้เฉพาะด้าน
            สามารถให้คำแนะนำเกี่ยวกับสินค้าคอมพิวเตอร์ อุปกรณ์ฮาร์ดแวร์ และการประกอบเครื่อง
            ตอบคำถามเกี่ยวกับราคา สเปค และความเหมาะสมของสินค้าต่างๆ
            ให้ตอบเป็นภาษาไทยที่สุภาพ เป็นมิตร และเข้าใจง่าย
</instructions>

<product_details>
สินค้าที่มีจำหน่าย:
- Gaming PC AMD Ryzen 5 GTX 1660 Super: 35,000 บาท (คลัง: 5 ชิ้น)
- โน้ตบุ๊ก ASUS VivoBook 15 i5-1135G7: 18,500 บาท (คลัง: 8 ชิ้น)
- Gaming PC Intel i7 RTX 3060: 45,000 บาท (คลัง: 3 ชิ้น)
- Office PC Intel i3 Integrated Graphics: 12,000 บาท (คลัง: 12 ชิ้น)
- CPU Intel Core i5-12400F: 6,500 บาท (คลัง: 15 ชิ้น)
- CPU AMD Ryzen 7 5700X: 8,900 บาท (คลัง: 10 ชิ้น)
- VGA RTX 4060 Ti 16GB: 18,500 บาท (คลัง: 7 ชิ้น)
- VGA GTX 1660 Super 6GB: 7,500 บาท (คลัง: 9 ชิ้น)
- RAM DDR4 16GB (8x2) 3200MHz Corsair: 2,800 บาท (คลัง: 20 ชิ้น)
- RAM DDR5 32GB (16x2) 5600MHz G.Skill: 5,500 บาท (คลัง: 6 ชิ้น)
- SSD 1TB NVMe M.2 Samsung 980 PRO: 3,200 บาท (คลัง: 18 ชิ้น)
- HDD 2TB Western Digital Blue: 1,800 บาท (คลัง: 14 ชิ้น)
- คีย์บอร์ด Logitech G Pro X Mechanical: 4,200 บาท (คลัง: 11 ชิ้น)
- เมาส์ Razer DeathAdder V3: 2,100 บาท (คลัง: 16 ชิ้น)
- จอมอนิเตอร์ 24 นิ้ว AOC 144Hz Gaming: 6,800 บาท (คลัง: 4 ชิ้น)
- จอมอนิเตอร์ 27 นิ้ว LG UltraGear 165Hz: 9,500 บาท (คลัง: 6 ชิ้น)
- เคสคอม NZXT H510 Mid Tower: 2,800 บาท (คลัง: 8 ชิ้น)
- PSU 650W 80+ Gold Seasonic: 3,200 บาท (คลัง: 13 ชิ้น)
- คูลเลอร์ CPU Noctua NH-D15: 2,900 บาท (คลัง: 7 ชิ้น)
- ลำโพง Logitech Z313 2.1 Speaker: 1,200 บาท (คลัง: 22 ชิ้น)
</product_details>

<long_term_memory>
ข้อมูลประวัติสำคัญของผู้ใช้:
- EventType.REQUEST: คอมอะไรก็ได้ 2 เครทรื่องครับ
  (ความต้องการ: User wants to buy two computers.)
- EventType.TRANSACTION: ซื้อคอม 2 ครับ
  (ความต้องการ: User wants to buy two computers.)
- EventType.REQUEST: อยากได้คอม
  (ความต้องการ: User wants to acquire a computer.)

ใช้ข้อมูลประวัติข้างต้นในการให้คำแนะนำที่เหมาะสม
</long_term_memory>
2. [HUMAN] สวัสดี
============================================================

============================================================
#### Prompt System
============================================================
เหตุผลที่ Enterprise ใช้แบบนี้ 🏢

  1. Production-Grade Requirements

  # Microsoft ต้องการผลลัพธ์ที่
  - Deterministic (ผลลัพธ์เหมือนเดิม)
  - Traceable (ตรวจสอบได้)
  - Consistent (รูปแบบเดียวกัน)
  - Scalable (ใช้ได้กับข้อมูลใหญ่)

  2. Complex Domain Logic

  # GraphRAG ต้องจัดการ
  - Knowledge graphs
  - Community detection
  - Multi-hop reasoning
  - Relationship extraction
  → JSON schema ไม่เพียงพอ สำหรับ logic ซับซ้อน

  3. Cross-Model Compatibility

  # Microsoft ต้องรองรับหลาย LLM
  - OpenAI GPT-4
  - Azure OpenAI
  - Local models
  - Future models
  → String prompts work กับทุก model

  การเปรียบเทียบ Approaches 📊

  Structured Output (JSON Schema)

  # ✅ ดี: Simple tasks
  class SimpleIntent(BaseModel):
      intent: str
      confidence: float

  # ❌ ไม่ดี: Complex reasoning
  class ComplexKnowledgeGraph(BaseModel):
      # ซับซ้อนเกินไป!
      nodes: List[Node]
      relationships: List[Relationship]
      communities: List[Community]
      reasoning_steps: List[ReasoningStep]

  String-based Prompts (Microsoft style)

  # ✅ ดี: Complex reasoning tasks
  COMMUNITY_REPORT_PROMPT = """
  You are an AI assistant that helps a human analyst...
  # 50+ บรรทัดของ detailed instructions
  # Examples และ constraints
  """

  เมื่อไหร่ควรใช้แบบไหน 🎯

  ใช้ Structured Output เมื่อ:

  ✅ Simple classification
  ✅ Data extraction
  ✅ Form filling
  ✅ API responses
  ✅ เริ่มต้น prototype

  # Example
  classify_sentiment(text) → {"sentiment": "positive", "score": 0.8}

  ใช้ String-based Prompts เมื่อ:

  ✅ Complex reasoning
  ✅ Multi-step analysis
  ✅ Domain expertise
  ✅ Creative generation
  ✅ Enterprise applications

  # Example  
  generate_research_report(data) → detailed analysis with citations

  Best Practices จาก Microsoft 📝

  1. Detailed Instructions

  PROMPT = """
  You are an expert analyst specializing in...

  # Role definition
  # Step-by-step process  
  # Output format
  # Constraints
  # Examples
  """

  2. Output Format Control

  # Microsoft ใช้
  "Return output as a well-formed JSON-formatted string"
  # แทนที่จะบังคับใช้ schema

  3. Grounding Rules

  # Rules เช่น
  - "Support all claims with evidence"
  - "Limit to maximum 200 words"
  - "Reference only provided data"

  4. Robust Parsing

  def parse_community_report(raw_output: str):
      try:
          # Primary: JSON parsing
          return json.loads(raw_output)
      except:
          # Fallback: Regex extraction
          return extract_with_regex(raw_output)
      except:
          # Final fallback: Manual processing
          return manual_parse(raw_output)

  สรุป: เลือกใช้อย่างไร 🎯

  Simple Tasks → Structured Output

  user_query = "ฉันอยากซื้อรองเท้า"
  result = simple_intent_classifier(user_query)
  # → {"intent": "purchase", "entity": "shoes"}

  Complex Tasks → String Prompts

  document = "50-page research paper"
  result = generate_comprehensive_analysis(document)
  # → detailed multi-section report with reasoning

  Hybrid Approach (Best of Both)

  # ใช้ string prompt + structured validation
  complex_result = complex_analyzer(data)
  validated_result = StructuredModel.parse_obj(complex_result)

  ข้อสรุป: Microsoft ใช้แบบนี้เพราะ GraphRAG เป็น complex reasoning system ไม่ใช่ simple
  classification ดังนั้น approach ของคุณถูกต้องสำหรับ enterprise-grade applications ครับ!

  แค่ต้องมี robust parsing และ error handling ที่ดีเท่านั้น 💪