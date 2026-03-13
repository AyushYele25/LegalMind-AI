import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Setup ────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="embeddings")
collection = client.get_or_create_collection(name="legal_docs")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_HISTORY_TURNS = 10

# ── Retrieve relevant law chunks with metadata ───────────
def retrieve_context(query, top_k=5):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    sources = []

    for chunk, meta in zip(chunks, metadatas):
        # law_display is stored directly from ingest.py — always clean name
        law = (meta.get("law_display") or meta.get("law_name") or "Indian Law") if meta else "Indian Law"
        page = meta.get("page") if meta else None
        page_str = str(page) if page else "—"

        context_parts.append(f"[{law} — Page {page_str}]\n{chunk}")

        source_entry = f"📖 {law}, Page {page_str}"
        if source_entry not in sources:
            sources.append(source_entry)

    return "\n\n---\n\n".join(context_parts), sources


# ── Ask LegalMind AI (streaming) ─────────────────────────
def ask_legalmind_stream(user_query, conversation_history=None):
    """
    Generator — yields text tokens as they stream in.
    Final yield is a dict sentinel {"__history__": updated_history}.
    """
    if conversation_history is None:
        conversation_history = []

    context, sources = retrieve_context(user_query)

    system_prompt = f"""You are LegalMind AI, an expert Indian legal assistant.
Your job is to help common people understand their legal rights in simple language.

Rules:
- Always explain in simple, easy-to-understand English
- Always cite the relevant Section number from the law
- Be empathetic and supportive
- Suggest practical next steps (which court, what to do)
- Do NOT add a Sources or References section at the end — sources are added automatically
- End with: "⚠️ This is AI guidance only, not formal legal advice."

Relevant excerpts from Indian Law (use these to answer):
{context}
"""

    trimmed_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    messages = (
        [{"role": "system", "content": system_prompt}]
        + trimmed_history
        + [{"role": "user", "content": user_query}]
    )

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
        stream=True
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer += delta
            yield delta

    # Append sources footer (deduplicated)
    if sources:
        seen = []
        unique_sources = []
        for s in sources:
            # Deduplicate by law name only (ignore page differences)
            law_key = s.split(", Page")[0]
            if law_key not in seen:
                seen.append(law_key)
                unique_sources.append(s)
        sources_text = "\n\n**📚 Sources:**\n" + "\n".join(unique_sources)
        full_answer += sources_text
        yield sources_text

    # Final sentinel with updated history
    updated_history = trimmed_history + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": full_answer},
    ]
    yield {"__history__": updated_history}


# ── Non-streaming version (for CLI testing) ──────────────
def ask_legalmind(user_query, conversation_history=None):
    full_answer = ""
    updated_history = conversation_history or []
    for chunk in ask_legalmind_stream(user_query, conversation_history):
        if isinstance(chunk, dict) and "__history__" in chunk:
            updated_history = chunk["__history__"]
        else:
            full_answer += chunk
    return full_answer, updated_history


# ── CLI Test ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🏛️  LegalMind AI — Indian Legal Assistant")
    print("=" * 50)
    history = []
    test_questions = [
        "I bought a phone that stopped working after 2 days. What can I do?",
        "My employer hasn't paid salary for 2 months. What are my rights?",
        "Someone hacked my bank account. What can I do under IT Act?",
    ]
    for q in test_questions:
        print(f"\n❓ {q}")
        answer, history = ask_legalmind(q, history)
        print(f"⚖️  {answer}")
        print("-" * 50)
