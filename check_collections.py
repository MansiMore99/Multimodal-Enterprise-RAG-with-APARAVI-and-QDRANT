import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def check_collection(collection_name):
    """Check the contents of a collection."""
    try:
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"\nCollection '{collection_name}' info:")
        print(f"Points count: {collection_info.points_count}")
        
        # Get all points
        points = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10
        )[0]
        
        print("\nPoints in collection:")
        for point in points:
            print(f"\nPoint ID: {point.id}")
            print(f"Source: {point.payload.get('source', 'N/A')}")
            print(f"Content preview: {point.payload.get('content', '')[:200]}...")
            
    except Exception as e:
        print(f"Error checking collection {collection_name}: {str(e)}")

if __name__ == "__main__":
    # Check the home_repair collection
    check_collection("home_repair") 