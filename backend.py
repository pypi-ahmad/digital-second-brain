"""
backend.py
This module handles all the core logic for the Digital Second Brain application.
It includes:
- OCR (Optical Character Recognition) for handwriting.
- LLM Integration (Ollama, OpenAI, Anthropic, Gemini).
- Vector Database Management (ChromaDB) for storing and retrieving notes.
- Knowledge Graph Generation (NetworkX, PyVis).
"""
import ollama
import chromadb
from chromadb.utils import embedding_functions
import uuid
import datetime
from openai import OpenAI
import anthropic
import google.genai as genai
import os
import tempfile
from pyvis.network import Network

# --- CONFIGURATION ---
# Initialize ChromaDB client for persistent vector storage
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")

# Use EmbeddingGemma for vectors (Since you have it!)
# If this fails, we can fall back to a default sentence-transformer
# Embeddings convert text into numbers (vectors) so the computer can understand meaning.
try:
    EMBEDDING_FUNC = embedding_functions.OllamaEmbeddingFunction(
        model_name="embeddinggemma",
        url="http://localhost:11434"
    )
    # Test if it works immediately to trigger fallback if not
    EMBEDDING_FUNC(["test"]) 
except Exception as e:
    print(f"Warning: embeddinggemma not found or Ollama not running. Using default embeddings. Details: {e}")
    # Fallback to a standard model if Ollama is not available
    EMBEDDING_FUNC = embedding_functions.DefaultEmbeddingFunction()

COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name="notes_db",
    embedding_function=EMBEDDING_FUNC
)

# --- AI WRAPPER (BYOK Support) ---
def get_llm_response(prompt, system_prompt, provider="Local (Ollama)", model="glm-4.7-flash", api_key=None):
    """
    A unified wrapper to call different AI providers.
    
    Args:
        prompt (str): The user's input or query.
        system_prompt (str): Instructions for the AI's behavior.
        provider (str): The AI provider to use (Local, OpenAI, etc.).
        model (str): The specific model name to call.
        api_key (str): API key for cloud providers (optional for Local).
        
    Returns:
        str: The text response from the AI.
    """
    try:
        if provider == "Local (Ollama)":
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']

        elif provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        elif provider == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(f"{system_prompt}\n\n{prompt}")
            return response.text

        return f"Error: Unsupported provider '{provider}'"

    except Exception as e:
        return f"Error: {str(e)}"

# --- CORE FUNCTIONS ---

def ocr_handwriting(image_path):
    """
    Uses DeepSeek-OCR (via Ollama) to transcribe handwritten notes from an image.
    
    Args:
        image_path (str): The file path to the image.
        
    Returns:
        str: The transcribed text.
    """
    print(f"Reading {image_path}...")
    try:
        response = ollama.chat(
            model='deepseek-ocr',
            messages=[{'role': 'user', 'content': 'Transcribe this handwritten note exactly. Maintain layout.', 'images': [image_path]}]
        )
        return response['message']['content']
    except Exception as e:
        return f"OCR Error: {e}"

def process_and_index_note(image_path, provider, model, api_key):
    """
    Full pipeline: OCR -> Clean/Tag (LLM) -> Index (Vector DB).
    
    1. Reads handwriting from the image.
    2. Uses an LLM to fix typos, generate tags, and summarize.
    3. Saves the processed note into ChromaDB for searching.
    
    Returns:
        tuple: (clean_text, tags, summary)
    """
    # 1. OCR
    raw_text = ocr_handwriting(image_path)
    if isinstance(raw_text, str) and raw_text.startswith("OCR Error:"):
        return raw_text, "General", "Note"
    
    # 2. Clean & Tag (LLM)
    system_prompt = """
    You are a Personal Knowledge Assistant. 
    1. Fix minor typos in this OCR text (assume it's notes).
    2. Extract 3-5 keywords/tags.
    3. Summarize the main topic in 1 sentence.
    
    Output Format:
    CLEAN_TEXT: [The text]
    TAGS: [Tag1, Tag2, Tag3]
    SUMMARY: [One sentence summary]
    """
    
    processed = get_llm_response(raw_text, system_prompt, provider, model, api_key)
    
    # Parse the LLM output (Basic string parsing)
    clean_text = raw_text # Fallback
    tags = "General"
    summary = "Note"
    
    if processed and "CLEAN_TEXT:" in processed:
        parts = processed.split("TAGS:")
        clean_text = parts[0].replace("CLEAN_TEXT:", "").strip()
        if len(parts) > 1:
            meta_parts = parts[1].split("SUMMARY:")
            tags = meta_parts[0].strip()
            if len(meta_parts) > 1:
                summary = meta_parts[1].strip()
    elif processed:
         clean_text = processed # Fallback if format is not followed exactly

    # 3. Store in Vector DB
    note_id = str(uuid.uuid4())
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    COLLECTION.add(
        documents=[clean_text],
        metadatas=[{"date": current_date, "tags": tags, "summary": summary, "source": image_path}],
        ids=[note_id]
    )
    
    return clean_text, tags, summary

def search_notes(query, n_results=3):
    """
    Performs a semantic search on the vector database.
    Finds notes that mean the same thing as the query, even if words are different.
    
    Args:
        query (str): The search text.
        n_results (int): Number of matching notes to return.
        
    Returns:
        str: Formatted string of search results.
    """
    try:
        results = COLLECTION.query(
            query_texts=[query],
            n_results=n_results
        )
        
        output = []
        # Chroma returns lists of lists
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                date_value = meta.get('date', '')
                tags_value = meta.get('tags', '')
                summary_value = meta.get('summary', '')
                output.append(f"**Date:** {date_value}\n**Tags:** {tags_value}\n**Summary:** {summary_value}\n\n{doc}\n---")
                
        return "\n".join(output) if output else "No matching notes found."
    except Exception as e:
        return f"Search Error: {str(e)}"

def chat_with_notes(message, history, provider, model, api_key):
    """
    RAG (Retrieval-Augmented Generation) function.
    1. Searches for relevant notes based on the user's message.
    2. Feeds those notes to the LLM as context.
    3. Asks the LLM to answer the question using that context.
    """
    _ = history
    # 1. Search DB
    context = search_notes(message, n_results=3)
    
    # 2. Generate Answer
    system_prompt = f"""
    You are a 'Second Brain' assistant. Answer the user's question based ONLY on their notes below.
    If the answer isn't in the notes, say "I couldn't find that in your notes."
    
    CONTEXT (USER NOTES):
    {context}
    """
    
    return get_llm_response(message, system_prompt, provider, model, api_key)

# --- NEW: KNOWLEDGE GRAPH GENERATOR ---
def generate_knowledge_graph():
    """
    Generates an HTML interactive graph of notes and tags.
    Uses NetworkX to build the graph structure and PyVis to visualize it.
    
    Nodes: Individual Notes (Blue) and Tags (Cyan).
    Edges: Connections between Notes and their Tags.
    
    Returns:
        str: Raw HTML content of the interactive graph.
    """
    try:
        all_data = COLLECTION.get() # Fetch all notes
        ids = all_data['ids']
        metadatas = all_data['metadatas']
        
        if not ids:
            return "<div>No notes to visualize yet!</div>"

        # Initialize Graph
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        net.force_atlas_2based()
        
        # Create Nodes & Edges
        for i, note_id in enumerate(ids):
            meta = metadatas[i]
            # Handle potential missing keys or None values
            summary = meta.get('summary', 'No Summary')
            if not summary: summary = "No Summary"
            tags_str = meta.get('tags', 'General')
            if not tags_str: tags_str = "General"

            short_summary = summary[:20] + "..." if len(summary) > 20 else summary
            
            # Add Note Node
            net.add_node(note_id, label=short_summary, title=summary, color="#4facfe", shape="box")
            
            # Add Tag Nodes and Connect
            tags = [t.strip() for t in tags_str.split(',')]
            for tag in tags:
                tag_id = f"tag_{tag.lower()}"
                net.add_node(tag_id, label=tag, color="#00f2fe", shape="ellipse")
                net.add_edge(note_id, tag_id)

        # Save and return HTML
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix="graph_").name
        net.save_graph(output_path)
        
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        os.remove(output_path)
        return html
    except Exception as e:
        return f"<div>Error generating graph: {str(e)}</div>"
