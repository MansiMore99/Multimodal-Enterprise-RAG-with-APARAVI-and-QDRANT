# multimodal_rag_app.py
import os
import json
import gradio as gr
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

def load_env():
    """Load environment variables from .env file."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("Successfully loaded environment variables from .env file")
    except Exception as e:
        print(f"Error loading .env file: {str(e)}")
        raise

# Load environment variables
load_env()

# ------------------------- Configuration -------------------------
# OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Debug print (showing only first 10 and last 4 characters)
print(f"Using API key starting with: {openai_api_key[:10]}...{openai_api_key[-4:]}")

print("Initializing OpenAI client...")
openai_client = OpenAI(api_key=openai_api_key)
print("Successfully initialized OpenAI client")

# Test OpenAI connection
try:
    print("Testing OpenAI connection...")
    test_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("OpenAI connection test successful")
except Exception as e:
    print(f"Error testing OpenAI connection: {str(e)}")
    if hasattr(e, 'response'):
        print(f"Response status: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
    raise ValueError("Failed to connect to OpenAI. Please check your API key.")

# Qdrant Configuration
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")

print(f"Initializing Qdrant client with URL: {qdrant_url}")
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=30.0,
    prefer_grpc=False
)

# Test Qdrant connection
try:
    print("Testing Qdrant connection...")
    collections = qdrant_client.get_collections()
    print(f"Successfully connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"Error connecting to Qdrant: {str(e)}")
    raise ValueError("Failed to connect to Qdrant. Please check your URL and API key.")

# Model Configuration
openai_model = "gpt-3.5-turbo"
temperature = 0.7
max_tokens = 1000

# System and User Prompts
system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
If the context doesn't contain relevant information, say so. 
Always be polite and professional."""

user_prompt = """Context information is below:
{context}

Current conversation:
{history}

User question: {message}

Please provide a helpful response based on the context and conversation history."""

# ------------------------- Utilities -------------------------
def get_embedding(text):
    """Get embedding for text using OpenAI's API."""
    try:
        print(f"Getting embedding for text: {text[:100]}...")  # Log first 100 chars
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        print("Successfully got embedding")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise

# ------------------------- Hybrid Search -------------------------
def hybrid_search(query, collections):
    """Perform hybrid search across multiple collections."""
    try:
        print(f"Performing hybrid search for query: {query}")
        print(f"Searching in collections: {collections}")
        
        # Get embeddings for the query
        try:
            query_embedding = get_embedding(query)
        except Exception as e:
            print(f"Failed to get embedding: {str(e)}")
            return []
        
        # Search in each collection
        all_matches = []
        for collection in collections:
            try:
                print(f"Searching in collection: {collection}")
                search_result = qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    limit=5,
                    score_threshold=0.05  # Even lower threshold to get more results
                )
                print(f"Found {len(search_result)} matches in {collection}")
                if search_result:
                    print(f"First match score: {search_result[0].score}")
                    print(f"First match content preview: {search_result[0].payload.get('content', '')[:200]}...")
                all_matches.extend(search_result)
            except Exception as e:
                print(f"Error searching collection {collection}: {str(e)}")
                continue
        
        # Sort all matches by score
        all_matches.sort(key=lambda x: x.score, reverse=True)
        print(f"Total matches found: {len(all_matches)}")
        if all_matches:
            print(f"Best match score: {all_matches[0].score}")
            print(f"Best match content preview: {all_matches[0].payload.get('content', '')[:200]}...")
        return all_matches[:5]  # Return top 5 matches across all collections
        
    except Exception as e:
        print(f"Error in hybrid_search: {str(e)}")
        return []

# ------------------------- Prediction Logic -------------------------
def predict(query, history, collections):
    if not collections:
        return "Please select at least one collection."

    try:
        matches = hybrid_search(query, collections)

        if not matches:
            return "No relevant data found in selected collections."

        # Print all matches for debugging
        print("\nAll matches found:")
        for i, match in enumerate(matches):
            print(f"\nMatch {i+1}:")
            print(f"Score: {match.score}")
            print(f"Source: {match.payload.get('source', 'N/A')}")
            print(f"Content preview: {match.payload.get('content', '')[:200]}...")

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
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return f"An error occurred: {str(e)}"

# ------------------------- Gradio UI -------------------------
def get_collections():
    try:
        print("Fetching collections from Qdrant...")
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        print(f"Found collections: {collection_names}")
        return collection_names
    except UnexpectedResponse as e:
        print(f"Qdrant API error getting collections: {str(e)}")
        print(f"Response content: {e.response.content}")
        return []
    except Exception as e:
        print(f"Error getting collections: {str(e)}")
        return []

def chat_ui(query, history, collections):
    if not collections:
        return "Please select at least one collection."

    try:
        print(f"\nProcessing query: {query}")
        print(f"Selected collections: {collections}")
        
        matches = hybrid_search(query, collections)

        if not matches:
            print("No matches found in hybrid search")
            return "I couldn't find any relevant information in the selected collections. Please try rephrasing your question or selecting different collections."

        # Format the context from matches
        context_parts = []
        for hit in matches:
            try:
                # Access the payload safely
                payload = hit.payload if hasattr(hit, 'payload') else {}
                content = payload.get('content', '')
                score = hit.score if hasattr(hit, 'score') else 0
                print(f"Match score: {score}")
                context_parts.append(f"Content: {content}\nScore: {score}")
            except Exception as e:
                print(f"Error processing match: {str(e)}")
                continue

        context = "\n".join(context_parts)
        print(f"Context length: {len(context)} characters")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(context=context, message=query, history=history)}
        ]

        print("Sending request to OpenAI...")
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Received response from OpenAI")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat_ui: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return f"I encountered an error while processing your request. Please try again or contact support if the issue persists."

def user_interaction(message, history, collection_names, current_collections):
    if not message:
        return history, current_collections

    try:
        # Convert history to the new message format
        formatted_history = []
        for msg in history:
            if isinstance(msg, tuple):
                role, content = msg
                formatted_history.append({"role": role.lower(), "content": content})
            elif isinstance(msg, dict):
                formatted_history.append(msg)

        user_message = {"role": "user", "content": message}
        bot_response = chat_ui(message, formatted_history, collection_names)
        bot_message = {"role": "assistant", "content": bot_response}
        
        formatted_history.append(user_message)
        formatted_history.append(bot_message)
        return formatted_history, current_collections
    except Exception as e:
        print(f"Error in user_interaction: {str(e)}")
        error_message = {"role": "assistant", "content": f"An error occurred: {str(e)}"}
        return history + [error_message], current_collections

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
                type="messages"
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