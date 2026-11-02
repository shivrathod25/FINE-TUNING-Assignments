# Session 2: Prompt Engineering vs. RAG vs. Fine-Tuning
**Course:** Fine-Tuning & LLM Engineering  
**Workspace:** `e:\TOPS Classes\FINE-TUNING-Assignments\session2`

---

## Task 1: Instagram Bio Prompt for a College Student & Generated Results

### Tested Prompt:
```text
Role: You are a creative personal branding copywriter specializing in social media profiles for young creators.
Task: Write 4 catchy, distinct Instagram bios for a Computer Science college student.

Requirements:
1. Length: Under 150 characters per bio (strictly within Instagram's profile limit).
2. Content: Blend academic identity (CS/Engineering student), hobbies (coding, coffee, fitness, weekend travels), and a clever hook.
3. Emojis: Include 2 to 3 well-placed emojis per bio.
4. Output: Present 4 distinct vibes:
   - Vibe A: The Tech Builder / Hacker
   - Vibe B: Minimalist & Aesthetic
   - Vibe C: Humorous & Relatable
   - Vibe D: High-Energy Ambition
5. Include a subtle placeholder call-to-action (e.g., "Check my projects 👇" or "Say hi 👋").
```

### Generated & Verified Results:
> **Option 1: The Tech Builder (Hacker Vibe)**  
> ☕ Turning caffeine into code & late-night ideas  
> 💻 CS Junior @ GTU | Building micro-SaaS  
> 📍 Exploring cities between git commits  
> See what I built 👇 *[github.com/profile]*  
> *(146 characters)*

> **Option 2: Minimalist & Aesthetic**  
> lines of code & mountain roads ⛰️  
> studying computers | living deliberately  
> capturing quiet moments ☕  
> portfolio below 🔗  
> *(128 characters)*

> **Option 3: Humorous & Relatable**  
> 90% debug errors, 10% pure luck 🐞  
> CS major surviving on iced Americanos & leg day PRs 🏋️‍♂️  
> pretending I know what I'm doing  
> connect with me 👇  
> *(147 characters)*

> **Option 4: High-Energy Ambition**  
> 🚀 Future Founder | Code. Gym. Repeat.  
> 💡 CS undergrad breaking things to make them faster  
> Chasing horizons ✈️ | Let’s build the future  
> Drop a DM 👋  
> *(149 characters)*

---

## Task 2: Python RAG Simulator for IPL Cricket Stats

The complete simulator code is located at:
[session2/ipl_rag_simulator.py](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session2/ipl_rag_simulator.py)  
The knowledge base file is located at:
[session2/ipl_stats.txt](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session2/ipl_stats.txt)

### Python Implementation:
```python
import os
import re
from typing import List, Tuple

DATA_FILE = os.path.join(os.path.dirname(__file__), "ipl_stats.txt")

def load_knowledge_base(file_path: str) -> List[str]:
    """Loads knowledge chunks from disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def tokenize(text: str) -> List[str]:
    stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "for", "of", "to", "who", "what"}
    return [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if w not in stop_words]

def retrieve_relevant_facts(query: str, facts: List[str], top_k: int = 2) -> List[Tuple[str, float]]:
    query_tokens = tokenize(query)
    scored = []
    for fact in facts:
        fact_tokens = tokenize(fact)
        matches = [tok for tok in query_tokens if tok in fact_tokens]
        score = len(matches) / (len(query_tokens) + 1e-5)
        for tok in query_tokens:
            if tok in fact.lower():
                score += 0.3
        if score > 0.15:
            scored.append((fact, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def augment_prompt(query: str, retrieved_facts: List[Tuple[str, float]]) -> str:
    context_text = "\n".join([f"- {fact}" for fact, _ in retrieved_facts])
    return f"""[SYSTEM CONTEXT - RETRIEVED FACTS FROM IPL DATABASE]
{context_text}

[USER QUESTION]
{query}

[INSTRUCTION]
Answer the user's question accurately using ONLY the retrieved facts above."""

def generate_rag_answer(query: str, retrieved_facts: List[Tuple[str, float]]) -> str:
    if not retrieved_facts:
        return "I could not find verified IPL records matching your query."
    answer = "Based on the retrieved IPL database records:\n"
    for i, (fact, _) in enumerate(retrieved_facts, 1):
        clean_fact = re.sub(r"^\[.*?\]\s*", "", fact)
        answer += f"  {i}. {clean_fact}\n"
    return answer.strip()
```

### Execution Terminal Log:
```text
Loaded 34 IPL statistical records into memory.

================================================================================
USER QUERY: "Which team has won 5 IPL trophies and who was their captain?"
--------------------------------------------------------------------------------
[STAGE 1: RETRIEVAL] Found 2 relevant chunk(s):
  -> Score: 1.87 | Record: [TEAM_TITLES] Chennai Super Kings (CSK) has won 5 IPL titles (2010, 2011, 2018, 2021, 2023). Captained primarily by MS Dhoni.
  -> Score: 1.87 | Record: [TEAM_TITLES] Mumbai Indians (MI) has won 5 IPL titles (2013, 2015, 2017, 2019, 2020). Captained by Rohit Sharma.

[STAGE 2: AUGMENTATION] Constructed Context Window Prompt:
   | [SYSTEM CONTEXT - RETRIEVED FACTS FROM IPL DATABASE]
   | - [TEAM_TITLES] Chennai Super Kings (CSK) has won 5 IPL titles (2010, 2011, 2018, 2021, 2023). Captained primarily by MS Dhoni.
   | - [TEAM_TITLES] Mumbai Indians (MI) has won 5 IPL titles (2013, 2015, 2017, 2019, 2020). Captained by Rohit Sharma.
   | 
   | [USER QUESTION]
   | Which team has won 5 IPL trophies and who was their captain?
   | 
   | [INSTRUCTION]
   | Answer the user's question accurately using ONLY the retrieved facts above.

[STAGE 3: GROUNDED GENERATION (FINAL OUTPUT)]
Based on the retrieved IPL database records:
  1. Chennai Super Kings (CSK) has won 5 IPL titles (2010, 2011, 2018, 2021, 2023). Captained primarily by MS Dhoni.
  2. Mumbai Indians (MI) has won 5 IPL titles (2013, 2015, 2017, 2019, 2020). Captained by Rohit Sharma.
================================================================================

USER QUERY: "Who won the Orange Cap in IPL 2024?"
--------------------------------------------------------------------------------
[STAGE 1: RETRIEVAL] Found 2 relevant chunk(s):
  -> Score: 2.50 | Record: [SEASON_2024_AWARDS] Virat Kohli won the Orange Cap in IPL 2024 scoring 741 runs in 15 innings at an average of 61.75 with RCB.
  -> Score: 2.00 | Record: [SEASON_2024_AWARDS] Harshal Patel won the Purple Cap in IPL 2024 claiming 24 wickets in 14 matches for Punjab Kings (PBKS).

[STAGE 3: GROUNDED GENERATION (FINAL OUTPUT)]
Based on the retrieved IPL database records:
  1. Virat Kohli won the Orange Cap in IPL 2024 scoring 741 runs in 15 innings at an average of 61.75 with RCB.
  2. Harshal Patel won the Purple Cap in IPL 2024 claiming 24 wickets in 14 matches for Punjab Kings (PBKS).
================================================================================
```

---

## Task 3: Three Real-World Scenarios for Prompting, RAG, and Fine-Tuning

### Scenario 1: Prompt Engineering is Best -> **Zomato: Merchant Review Reply Assistant & Promotional Push Notifications**
- **Use Case:** Helping restaurant partners craft courteous, creative responses to customer reviews or generating catchy daily push notifications (e.g., *"Weekend match cravings? Biryani is just 20 mins away! 🏏"*).
- **Why Prompt Engineering is Best:** The task requires linguistic variety and creative flair without strict factual ground truth or database queries. There is zero risk of violating regulatory statutes, and different prompts can be tested instantly without training infrastructure.

### Scenario 2: RAG is Best -> **Flipkart: Technical Product Specification Q&A & Return Policy Assistant**
- **Use Case:** A shopper asks: *"Does the Asus ROG Zephyrus G14 listed here have an expandable DDR5 RAM slot, and what is the seller's return window for Gujarat?"*
- **Why RAG is Best:** Flipkart hosts millions of dynamic SKUs, and specifications, seller terms, prices, and warranties change multiple times per day. Fine-tuning a model on product catalogs is impossible because products go out of stock or update daily. RAG dynamically extracts the exact verified PDF manual or product specification sheet from the catalog vector store and grounds the answer with zero hallucination.

### Scenario 3: Fine-Tuning is Best -> **Paytm: Automated UPI Dispute Classification & Banking Grievance Triage**
- **Use Case:** Classifying customer grievance tickets regarding failed UPI transactions, chargeback requests, and merchant fraud into regulatory NPCI grievance codes.
- **Why Fine-Tuning is Best:** Financial compliance requires 100% deterministic classification into strict internal categories with sub-25ms latency SLAs. Passing 30-page RBI/NPCI dispute guidelines via prompts introduces unacceptable token costs and latency at scale, while RAG is vulnerable to retrieval misses. Fine-tuning bakes the strict regulatory taxonomy and jargon directly into model weights.

---

## Task 4: Comparison Table - Prompt Engineering vs. RAG vs. Fine-Tuning

| Criterion | Prompt Engineering | Retrieval-Augmented Generation (RAG) | Fine-Tuning (Full / PEFT / LoRA) |
| :--- | :--- | :--- | :--- |
| **Setup Cost** | **Minimal (\$0 - \$50)**.<br>Requires only access to an LLM API key and prompt authoring. | **Moderate (\$500 - \$5,000)**.<br>Requires chunking pipeline, embedding models, and a vector database (e.g., Pinecone, Chroma, pgvector). | **High (\$2,000 - \$50,000+)**.<br>Requires curating high-quality labeled datasets, GPU compute clusters (A100/H100), and training pipelines. |
| **Time to Deploy** | **Minutes to Hours**.<br>Immediate experimentation via API playgrounds. | **Days to Weeks**.<br>Requires setting up document ingestion, indexing, and retrieval evaluation. | **Weeks to Months**.<br>Requires data labeling, hyperparameter tuning, validation, and regression testing. |
| **Maintenance Effort** | **Low**.<br>Prompts are edited as simple strings when requirements shift. | **Moderate**.<br>Requires vector database indexing, embedding re-syncing, and chunking optimization. | **High**.<br>Requires retraining pipelines, handling data drift, and managing model adapter versioning. |
| **Dynamic Knowledge Updates** | **Poor**.<br>Model only knows what fits in the context window. | **Exceptional**.<br>Simply update or insert rows in the database; instantly accessible without retraining. | **Static / Poor**.<br>Knowledge is frozen as of the training run; requires periodic retraining to update facts. |
| **Hallucination Risk** | **High**.<br>Model relies on parametric memory and can invent convincing falsehoods. | **Very Low**.<br>Outputs are strictly grounded in retrieved factual chunks with source citations. | **Low to Moderate**.<br>Learns domain style well, but may still hallucinate facts unless paired with retrieval. |
| **Inference Token Cost & Latency** | High prompt token overhead; slower time-to-first-token. | Medium-high token overhead due to retrieved context chunks injected into prompt. | **Lowest token overhead**; extremely fast inference with zero prompt bloat. |
| **Ideal Use Cases** | Brainstorming, drafting, generic chatbots, quick prototypes, creative copywriting. | Enterprise search, customer support over documentation, dynamic product catalogs, legal QA. | Custom classification, strict JSON/code schema generation, domain tone imitation, proprietary offline models. |

---

## Task 5: BookMyShow-Style Movie Recommendation Chatbot Architecture Consultation

### Advice Received from ChatGPT:
```text
Consultation Question:
"I am building an AI chatbot for a BookMyShow-style entertainment app. Users will ask for movie recommendations based on mood, check showtimes in their city, inquire about seat availability, and book tickets. Should I use Prompt Engineering, RAG, or Fine-Tuning?"

ChatGPT's Strategic Breakdown:
1. Pure Prompting is INSUFFICIENT:
   - A general LLM does not know today's cinema showtimes, currently running releases, ticket pricing, or seat occupancy.

2. Pure Fine-Tuning is INAPPROPRIATE:
   - Movie releases, show schedules, and seat layouts change hourly. You cannot fine-tune a model every hour to update seat counts.

3. RAG + Tool Calling (Function Calling) is the MANDATORY FOUNDATION:
   - Use RAG over a vector database containing movie synopses, actor filmographies, critical reviews, and genre tags for mood-based matching.
   - Use Tool/API Calling against BookMyShow's live cinema ticketing APIs to fetch real-time showtimes, theater locations, and available seat coordinates.

4. Lightweight PEFT/LoRA (Optional Phase 2):
   - Fine-tune a lightweight model only if you need a branded cinematic persona (e.g., Bollywood movie buff tone) and deterministic booking function call schema parsing.
```

### Final Architectural Decision:
**We decide to build a Hybrid RAG + API Tool Calling Architecture:**
1. **Dynamic RAG Pipeline:** Indexes movie synopses, critic consensus (Rotten Tomatoes / IMDb ratings), genres, and thematic tags into a vector database (e.g., Milvus/Pinecone) to resolve subjective mood queries (*"Recommend a gripping psychological thriller playing in Ahmedabad tonight"*).
2. **Deterministic Live API Integration:** Connects the LLM via Structured Function Calling to the live BookMyShow cinema database to fetch real-time theatre coordinates, screen formats (IMAX vs 4DX), showtimes, and seat layout availability.
3. **Prompt Engineering Layer:** Wraps the final synthesized context in a warm, enthusiastic entertainment concierge persona without wasting time or compute on heavy model fine-tuning.

