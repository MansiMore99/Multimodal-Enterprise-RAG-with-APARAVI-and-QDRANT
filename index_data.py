import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def get_embedding(text):
    """Get embedding for text using OpenAI's API."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def format_content(content):
    """Format the content to be more search-friendly."""
    # Remove extra newlines and spaces
    content = ' '.join(content.split())
    
    # Format sink unclogging content
    if "How to Unclog a Sink" in content:
        steps = content.split('.')
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            if step.strip():
                formatted_steps.append(f"Step {i}: {step.strip()}")
        return "How to Unclog a Sink:\n" + "\n".join(formatted_steps)
    
    # Format faucet repair content
    if "Finding and Repairing Faucet Leaks" in content:
        # Split into sections
        sections = content.split("1. Finding and Repairing Faucet Leaks")
        if len(sections) > 1:
            tips = sections[0].strip()
            main_content = sections[1].strip()
            
            # Format the content
            formatted = "How to Repair Faucet Leaks:\n\n"
            formatted += "Important Tips:\n"
            formatted += tips.replace("•", "• ").replace("  ", " ") + "\n\n"
            
            # Add main content
            formatted += "Repair Instructions:\n"
            formatted += main_content.replace("  ", " ")
            
            return formatted
    
    return content

def index_file(file_path, collection_name, point_id):
    """Index a text file into Qdrant."""
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Format the content
    formatted_content = format_content(content)
    
    # Create collection if it doesn't exist
    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    
    # Get embedding for the content
    embedding = get_embedding(formatted_content)
    
    # Create point
    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "content": formatted_content,
            "source": file_path
        }
    )
    
    # Upload point
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[point]
    )
    
    print(f"Successfully indexed {file_path} into collection {collection_name}")
    print(f"Formatted content preview: {formatted_content[:200]}...")

if __name__ == "__main__":
    # Index both files into the home_repair collection
    index_file("rag_Data/Unclog a Sink.txt", "home_repair", 1)
    index_file("rag_Data/faucet-leaks-and-repair-guide.txt", "home_repair", 2) 