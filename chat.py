# multimodal_rag_app.py
import os
import json
import gradio as gr
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest

# ------------------------- Configuration -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, 'config.json')) as f:
    config = json.load(f)

with open(os.path.join(current_dir, 'llm_settings.json')) as f:
    llm_settings = json.load(f)

# API keys & endpoints
qdrant_url = config["qdrant_url"]
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=config["qdrant_api_key"]
)
openai_api_key  = llm_settings["openai_api_key"]
openai_model    = llm_settings["openai_model"]
system_prompt   = llm_settings["system_prompt"]
user_prompt     = llm_settings["user_prompt"]
temperature     = llm_settings["temperature"]
max_tokens      = llm_settings["max_tokens"]
embedding_model = llm_settings.get("embedding_model", "text-embedding-3-small")

# Qdrant & OpenAI setup
qdrant_client = QdrantClient(url=qdrant_url)
openai_client = OpenAI(api_key=openai_api_key)

# ------------------------- Utilities -------------------------
def get_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding

# ------------------------- Hybrid Search -------------------------
def hybrid_search(query, collections):
    query_vector = get_embedding(query)
    all_matches = []

    for collection in collections:
        search_result = qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=5
        )
        all_matches.extend(search_result)

    return all_matches

# ------------------------- Prediction Logic -------------------------
def predict(query, history, collections):
    if not collections:
        return "Please select at least one collection."

    matches = hybrid_search(query, collections)

    if not matches:
        return "No relevant data found in selected collections."

    context = "\n".join([
        f"Content: {hit.payload.get('content', '')}\nScore: {hit.score}" for hit in matches
    ])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt.format(context=context, message=query, history=history)}
    ]

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# ------------------------- Gradio UI -------------------------
def get_collections():
    try:
        return [collection.name for collection in qdrant_client.get_collections().collections]
    except Exception as e:
        return [f"Error: {str(e)}"]

def chat_ui(query, collections, history=[]):
    return predict(query, history, collections)

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    min-height: 100vh;
    padding: 20px;
}
.container {
    max-width: 1000px;
    margin: auto;
    padding-top: 1.5rem;
}
#component-0 {
    max-width: 1000px;
    margin: auto;
}
.gradio-interface {
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.1);
}
.chat-message {
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    animation: fadeIn 0.5s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.chat-message.user {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196f3;
}
.chat-message.bot {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 4px solid #9c27b0;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin-right: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.chat-message .message {
    flex: 1;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Header styling */
h1 {
    color: #1a237e;
    text-align: center;
    margin-bottom: 1rem;
    font-size: 2.5rem !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}
h3 {
    color: #303f9f;
    text-align: center;
    margin-bottom: 2rem;
}

/* Button styling */
button {
    background: linear-gradient(135deg, #4a148c 0%, #7b1fa2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}

/* Input styling */
input[type="text"], textarea {
    border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #7b1fa2 !important;
    box-shadow: 0 0 0 2px rgba(123, 31, 162, 0.2) !important;
}

/* Dropdown styling */
select {
    border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
    background: white !important;
}

/* Tips section styling */
.tips-section {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 4px solid #4caf50;
}

/* Collection dropdown container */
.collection-container {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 4px solid #ff9800;
    margin-bottom: 1.5rem;
}

/* Refresh button */
.refresh-button {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%) !important;
}

/* Send button */
.send-button {
    background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%) !important;
}
"""

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("""
    # 🤖 Multimodal RAG Chat Interface
    ### Powered by Qdrant + OpenAI
    
    Select your collections and start chatting with the AI assistant!
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="collection-container"):
                collection_dropdown = gr.Dropdown(
                    choices=get_collections(),
                    label="📚 Select Collection(s)",
                    multiselect=True,
                    info="Choose one or more collections to search through"
                )
        with gr.Column(scale=1):
            refresh_button = gr.Button("🔄 Refresh Collections", variant="secondary", elem_classes="refresh-button")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=600,
                show_copy_button=True,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False
            )
        with gr.Column(scale=1):
            with gr.Group(elem_classes="tips-section"):
                gr.Markdown("""
                ### 💡 Tips
                - Select multiple collections for broader search
                - Be specific in your questions
                - Use natural language
                """)

    with gr.Row():
        message_input = gr.Textbox(
            label="Type your question",
            placeholder="Ask anything about the selected collections...",
            lines=3,
            max_lines=5
        )
        send_button = gr.Button("Send", variant="primary", size="lg", elem_classes="send-button")

    current_collections = gr.State([])

    def user_interaction(message, history, collection_names, current_collections):
        if not message:
            return history, current_collections

        user_message = ("You", message)
        bot_message = ("Assistant", chat_ui(message, collection_names, history + [user_message]))
        history.append(user_message)
        history.append(bot_message)
        return history, current_collections

    send_button.click(
        user_interaction,
        inputs=[message_input, chatbot, collection_dropdown, current_collections],
        outputs=[chatbot, current_collections]
    ).then(
        lambda: "",  # Clear the input box
        None,
        message_input
    )

    collection_dropdown.change(
        lambda: ([], []),
        outputs=[chatbot, current_collections]
    )

    refresh_button.click(get_collections, outputs=[collection_dropdown])

    # Add keyboard shortcut for sending messages
    message_input.submit(
        user_interaction,
        inputs=[message_input, chatbot, collection_dropdown, current_collections],
        outputs=[chatbot, current_collections]
    ).then(
        lambda: "",  # Clear the input box
        None,
        message_input
    )

interface.launch(share=True)