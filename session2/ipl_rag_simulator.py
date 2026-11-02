"""
IPL Cricket Stats - RAG (Retrieval-Augmented Generation) Simulator
===================================================================
Assignment Session 2 - Task 2
Simulates an end-to-end RAG pipeline using file reading, keyword/string matching,
context augmentation, and response generation grounded in retrieved facts.
"""

import os
import re
from typing import List, Tuple

DATA_FILE = os.path.join(os.path.dirname(__file__), "ipl_stats.txt")


def load_knowledge_base(file_path: str) -> List[str]:
    """Step 1: Load and parse the raw knowledge base document into distinct chunk lines."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge base file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines


def tokenize(text: str) -> List[str]:
    """Helper to convert query or text into clean lower-case tokens."""
    stop_words = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "for", 
        "of", "to", "who", "what", "how", "many", "has", "have", "with", "by", 
        "i", "me", "my", "tell", "about", "give", "please", "can", "you", "does"
    }
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [w for w in words if w not in stop_words]


def retrieve_relevant_facts(query: str, facts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Step 2: Retrieval Engine.
    Searches candidate lines using token matching and keyword weighting.
    Returns the top_k most relevant facts with similarity scores.
    """
    query_tokens = tokenize(query)
    scored_facts = []

    for fact in facts:
        fact_tokens = tokenize(fact)
        if not fact_tokens:
            continue
        
        # Calculate overlap score
        matches = [tok for tok in query_tokens if tok in fact_tokens]
        
        # Bonus weighting for exact numerical or proper noun matches (e.g. '2024', 'csk', 'kohli')
        score = len(matches) / (len(query_tokens) + 1e-5)
        
        # Direct substring match boost
        for tok in query_tokens:
            if tok in fact.lower():
                score += 0.3
                
        if score > 0.15:
            scored_facts.append((fact, score))
            
    # Sort descending by relevance score
    scored_facts.sort(key=lambda x: x[1], reverse=True)
    return scored_facts[:top_k]


def augment_prompt(query: str, retrieved_facts: List[Tuple[str, float]]) -> str:
    """
    Step 3: Augmentation.
    Combines the user query with the retrieved factual knowledge.
    """
    context_text = "\n".join([f"- {fact}" for fact, score in retrieved_facts])
    
    augmented_prompt = f"""[SYSTEM CONTEXT - RETRIEVED FACTS FROM IPL DATABASE]
{context_text if context_text else "No relevant records found in local database."}

[USER QUESTION]
{query}

[INSTRUCTION]
Answer the user's question accurately using ONLY the retrieved facts above.
If the facts do not contain the answer, state that the database lacks this record.
"""
    return augmented_prompt


def generate_rag_answer(query: str, retrieved_facts: List[Tuple[str, float]]) -> str:
    """
    Step 4: Generation Engine (Simulated Grounded Synthesis).
    Generates a clear natural language response synthesized directly from retrieved records.
    """
    if not retrieved_facts:
        return (
            "I could not find verified IPL statistics matching your question in the local database. "
            "Please check player names or record categories."
        )

    clean_facts = [fact for fact, _ in retrieved_facts]
    
    # Synthesis presentation
    answer = "Based on the retrieved IPL database records:\n"
    for i, fact in enumerate(clean_facts, 1):
        # Strip internal tags like [TEAM_TITLES] for friendly output
        readable_fact = re.sub(r"^\[.*?\]\s*", "", fact)
        answer += f"  {i}. {readable_fact}\n"
        
    return answer.strip()


def run_rag_pipeline(user_query: str, facts: List[str]):
    """Orchestrates the entire RAG pipeline and prints a clean step-by-step breakdown."""
    print("=" * 80)
    print(f"USER QUERY: \"{user_query}\"")
    print("-" * 80)
    
    # 1. Retrieve
    retrieved = retrieve_relevant_facts(user_query, facts, top_k=2)
    print(f"[STAGE 1: RETRIEVAL] Found {len(retrieved)} relevant chunk(s):")
    for fact, score in retrieved:
        print(f"  -> Score: {score:.2f} | Record: {fact}")
    print()
    
    # 2. Augment
    print("[STAGE 2: AUGMENTATION] Constructed Context Window Prompt:")
    augmented_prompt = augment_prompt(user_query, retrieved)
    for line in augmented_prompt.splitlines():
        print(f"   | {line}")
    print()
    
    # 3. Generate
    print("[STAGE 3: GROUNDED GENERATION (FINAL OUTPUT)]")
    final_output = generate_rag_answer(user_query, retrieved)
    print(final_output)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    facts_database = load_knowledge_base(DATA_FILE)
    print(f"Loaded {len(facts_database)} IPL statistical records into memory.\n")
    
    # Demonstration Queries for Assignment Submission
    test_queries = [
        "Which team has won 5 IPL trophies and who was their captain?",
        "Who scored the highest individual score in IPL history and for which team?",
        "Who won the Orange Cap in IPL 2024?",
        "Who is the all-time highest wicket-taker in the IPL?",
        "What is the highest team total scored in an IPL match?"
    ]
    
    for query in test_queries:
        run_rag_pipeline(query, facts_database)

