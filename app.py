import gradio as gr
from gradio import ChatMessage
from chatbot import ask_legalmind_stream

# ── Custom CSS ────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold: #C9A84C; --gold-light: #E8C97A; --dark: #0D0D0D;
    --dark-2: #141414; --dark-3: #1C1C1C; --dark-4: #252525;
    --text: #E8E8E0; --text-muted: #888880; --border: rgba(201,168,76,0.15);
}
* { box-sizing: border-box; }
body, .gradio-container { background: var(--dark) !important; font-family: 'DM Sans', sans-serif !important; color: var(--text) !important; }
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; padding: 0 !important; }
.hero-header { background: linear-gradient(135deg,#0D0D0D 0%,#1a1400 50%,#0D0D0D 100%); border-bottom: 1px solid var(--border); padding: 48px 40px 36px; text-align: center; position: relative; overflow: hidden; }
.hero-icon { font-size: 48px; display: block; margin-bottom: 16px; filter: drop-shadow(0 0 20px rgba(201,168,76,0.4)); animation: pulse-glow 3s ease-in-out infinite; }
@keyframes pulse-glow { 0%,100% { filter: drop-shadow(0 0 20px rgba(201,168,76,0.4)); } 50% { filter: drop-shadow(0 0 35px rgba(201,168,76,0.7)); } }
.hero-title { font-family: 'Playfair Display',serif !important; font-size: 52px !important; font-weight: 900 !important; color: var(--gold) !important; letter-spacing: -1px; line-height: 1.1; margin: 0 0 8px !important; }
.hero-subtitle { font-size: 15px !important; color: var(--text-muted) !important; font-weight: 300; letter-spacing: 3px; text-transform: uppercase; margin: 0 0 24px !important; }
.hero-badges { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.badge { background: rgba(201,168,76,0.08); border: 1px solid rgba(201,168,76,0.25); color: var(--gold-light); padding: 6px 16px; border-radius: 100px; font-size: 12px; font-weight: 500; }
.stats-bar { background: var(--dark-3); border: 1px solid var(--border); border-top: none; padding: 12px 24px; display: flex; gap: 32px; flex-wrap: wrap; }
.stat-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--gold); margin-right: 8px; animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
.stat-text { color: var(--text-muted) !important; font-size: 12px !important; }
#chatbot { background: var(--dark-2) !important; border: 1px solid var(--border) !important; }
#question-input textarea { background: var(--dark-4) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--text) !important; font-size: 15px !important; padding: 14px 18px !important; }
#question-input textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(201,168,76,0.1) !important; outline: none !important; }
#question-input label { color: var(--gold) !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; font-weight: 500 !important; }
#ask-btn { background: linear-gradient(135deg,var(--gold),#A8872E) !important; border: none !important; border-radius: 12px !important; color: var(--dark) !important; font-weight: 700 !important; font-size: 15px !important; transition: all 0.2s !important; }
#ask-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(201,168,76,0.35) !important; }
#clear-btn { background: transparent !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: var(--text-muted) !important; font-size: 13px !important; transition: all 0.2s !important; }
#clear-btn:hover { border-color: rgba(255,255,255,0.2) !important; color: var(--text) !important; }
.footer-bar { background: var(--dark); border: 1px solid var(--border); border-top: none; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
.footer-disclaimer { color: var(--text-muted) !important; font-size: 12px !important; }
.footer-version { color: rgba(201,168,76,0.5) !important; font-size: 11px !important; }
footer { display: none !important; }
"""

# ── Streaming Chat Function ───────────────────────────────
def chat(user_message, history, llm_history):
    if not user_message.strip():
        yield "", history, llm_history
        return

    # Show user message immediately
    history = history + [ChatMessage(role="user", content=user_message)]
    yield "", history, llm_history

    # Stream assistant response token by token
    assistant_text = ""
    updated_llm_history = llm_history
    history = history + [ChatMessage(role="assistant", content="▌")]

    try:
        for chunk in ask_legalmind_stream(user_message, llm_history):
            if isinstance(chunk, dict) and "__history__" in chunk:
                updated_llm_history = chunk["__history__"]
            else:
                assistant_text += chunk
                history[-1] = ChatMessage(role="assistant", content=assistant_text + "▌")
                yield "", history, updated_llm_history

        # Final — remove cursor
        history[-1] = ChatMessage(role="assistant", content=assistant_text)
        yield "", history, updated_llm_history

    except Exception as e:
        error_msg = (
            f"⚠️ **Error:** {str(e)}\n\n"
            "This could be due to:\n"
            "- Groq API rate limit — wait a moment and try again\n"
            "- Network issue — check your internet connection\n"
            "- API key issue — check your `.env` file"
        )
        history[-1] = ChatMessage(role="assistant", content=error_msg)
        yield "", history, updated_llm_history


# ── UI ────────────────────────────────────────────────────
with gr.Blocks(title="LegalMind AI — Indian Legal Assistant") as demo:

    gr.HTML("""
    <div class="hero-header">
        <span class="hero-icon">⚖️</span>
        <h1 class="hero-title">LegalMind AI</h1>
        <p class="hero-subtitle">India's Intelligent Legal Assistant</p>
        <div class="hero-badges">
            <span class="badge">⚡ Powered by LLaMA 3.3</span>
            <span class="badge">📚 14 Indian Laws</span>
            <span class="badge">🇮🇳 Built for India</span>
            <span class="badge">🔒 Free & Confidential</span>
        </div>
    </div>""")

    gr.HTML("""
    <div class="stats-bar">
        <span class="stat-text"><span class="stat-dot"></span>Live & Ready</span>
        <span class="stat-text">⚖️ 14 Indian Laws Indexed</span>
        <span class="stat-text">🗂 1,762 Legal Chunks</span>
        <span class="stat-text">⚡ RAG-Powered Retrieval</span>
        <span class="stat-text">📋 Cites Exact Sections & Pages</span>
        <span class="stat-text">🔄 Streaming Responses</span>
    </div>""")

    chatbot = gr.Chatbot(
        height=480,
        elem_id="chatbot",
        show_label=False,
        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=legalmind")
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Describe your legal problem... e.g. 'My employer hasn't paid salary for 2 months'",
            elem_id="question-input",
            label="YOUR QUESTION",
            lines=2,
            scale=5
        )
        ask_btn = gr.Button("Ask ⚖️", elem_id="ask-btn", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Conversation", elem_id="clear-btn")

    gr.Examples(
        examples=[
            ["I bought a phone that stopped working after 2 days. What are my rights?"],
            ["My employer hasn't paid my salary for 2 months. What can I do?"],
            ["How do I file an RTI application?"],
            ["Someone cheated me of money. What IPC section applies?"],
            ["I was in a road accident. What are my rights under Motor Vehicles Act?"],
            ["Someone hacked my bank account. What can I do under IT Act?"],
            ["My landlord is illegally evicting me. What are my rights?"],
            ["What protection does a woman have against domestic violence?"],
            ["A restaurant served me rotten food. What legal action can I take?"],
            ["My business partner broke our contract. What can I do?"],
        ],
        inputs=msg,
        label="💡 Sample Questions across All 14 Laws — Click to Try"
    )

    gr.HTML("""
    <div class="footer-bar">
        <span class="footer-disclaimer">⚠️ AI guidance only — not formal legal advice. Consult a qualified lawyer for serious matters.</span>
        <span class="footer-version">LEGALMIND AI v2.0</span>
    </div>""")

    llm_history = gr.State([])

    ask_btn.click(chat, [msg, chatbot, llm_history], [msg, chatbot, llm_history])
    msg.submit(chat, [msg, chatbot, llm_history], [msg, chatbot, llm_history])
    clear_btn.click(lambda: ("", [], []), outputs=[msg, chatbot, llm_history])

if __name__ == "__main__":
    demo.launch(share=False, css=custom_css)
