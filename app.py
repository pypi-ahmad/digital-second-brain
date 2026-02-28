"""
app.py
This file defines the User Interface (UI) for the Digital Second Brain.
It uses Gradio to create a web-based interface with tabs for:
1. Scanning & Indexing Notes
2. Searching Notes
3. Chatting with Notes (RAG)
4. Visualizing the Knowledge Graph
"""

import gradio as gr
import os
import tempfile
import backend

# --- UI LOGIC ---
# These functions act as bridges between the UI inputs and the backend logic.

def handle_scan(image, provider, model, key):
    """
    Handles the "Scan & Index" button click.
    Takes an image, processes it via backend, and returns results to the UI.
    """
    if image is None:
        return "Please upload an image.", "", ""
    
    # Save temp image for path usage
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix="temp_note_") as tmp:
        temp_path = tmp.name
    image.save(temp_path)
    
    try:
        text, tags, summary = backend.process_and_index_note(temp_path, provider, model, key)
    finally:
        os.remove(temp_path)
    return text, tags, summary

def handle_search(query):
    """
    Handles the "Find Notes" button click.
    Passes the search query to the backend.
    """
    return backend.search_notes(query)

def handle_chat(message, history, provider, model, key):
    """
    Handles the chat interactions.
    Called by the Gradio ChatInterface.
    """
    return backend.chat_with_notes(message, history, provider, model, key)

def handle_graph():
    """
    Handles the "Refresh Graph" button click.
    Requests the graph HTML from the backend.
    """
    return backend.generate_knowledge_graph()

# --- LAYOUT ---
# Defines the visual structure of the application.
with gr.Blocks(title="Second Brain 🧠") as app:
    gr.Markdown("# 🧠 Digital Second Brain: Handwritten Notes to Knowledge Base")
    
    # Sidebar for Settings: Allows user to choose AI models and providers.
    with gr.Sidebar():
        gr.Markdown("## ⚙️ Settings")
        provider_dd = gr.Dropdown(["Local (Ollama)", "OpenAI", "Anthropic", "Gemini"], label="AI Provider", value="Local (Ollama)")
        model_dd = gr.Dropdown(
            ["glm-4.7-flash", "lfm2.5-thinking", "llama3", "gpt-4o", "claude-3-5-sonnet"], 
            label="Model (Cleaner/Chat)", 
            value="glm-4.7-flash", 
            allow_custom_value=True
        )
        api_key_input = gr.Textbox(label="API Key (Cloud Only)", type="password")

    # Tabs: Organize different features into separate views.
    with gr.Tabs():
        
        # TAB 1: INGEST
        # Interface for uploading images and viewing processed results.
        with gr.TabItem("📸 Scan & Index"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload or Snap Photo", sources=["upload", "webcam"])
                    scan_btn = gr.Button("🚀 Convert & Save to Memory", variant="primary")
                with gr.Column():
                    output_text = gr.TextArea(label="Digitized Text", interactive=True)
                    with gr.Row():
                        tags_out = gr.Textbox(label="Auto-Tags")
                        summary_out = gr.Textbox(label="Summary")
            
            scan_btn.click(handle_scan, [img_input, provider_dd, model_dd, api_key_input], [output_text, tags_out, summary_out])

        # TAB 2: SEARCH
        with gr.TabItem("🔍 Semantic Search"):
            search_input = gr.Textbox(label="Search your notes (e.g., 'Project deadlines next week')")
            search_btn = gr.Button("Find Notes")
            search_results = gr.Markdown(label="Results")
            
            search_btn.click(handle_search, search_input, search_results)

        # TAB 3: CHAT (RAG)
        with gr.TabItem("💬 Chat with Notes"):
            gr.ChatInterface(
                fn=handle_chat, 
                additional_inputs=[provider_dd, model_dd, api_key_input],
                title="Ask your Second Brain",
                description="I will look through your handwritten notes to answer."
            )

        # TAB 4: GRAPH (NEW!)
        with gr.TabItem("🕸️ Knowledge Graph"):
            gr.Markdown("Visualizing connections between your notes and tags.")
            refresh_btn = gr.Button("🔄 Refresh Graph")
            graph_html = gr.HTML()
            refresh_btn.click(handle_graph, None, graph_html)

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
