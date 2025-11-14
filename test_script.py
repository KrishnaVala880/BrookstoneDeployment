"""
Test script to verify Pinecone + Ollama integration
Run this before deploying your main app
"""

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

INDEX_NAME = "brookstone-faq-ollama"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def test_ollama_embeddings():
    """Test Ollama embeddings"""
    print("\n" + "="*60)
    print("1. Testing Ollama Embeddings")
    print("="*60)
    
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Test embedding generation
        test_text = "What are the amenities at Brookstone?"
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Ollama embeddings working!")
        print(f"   Model: nomic-embed-text")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   Expected dimension: 768")
        
        if len(embedding) != 768:
            print(f"⚠️  WARNING: Dimension mismatch! Got {len(embedding)}, expected 768")
            return None
        
        print(f"✅ Dimension matches Pinecone index!")
        return embeddings
        
    except Exception as e:
        print(f"❌ Ollama embeddings failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Ollama is running: curl http://localhost:11434")
        print("2. Check if model exists: ollama list")
        print("3. Pull model if needed: ollama pull nomic-embed-text")
        return None


def test_pinecone_connection(embeddings):
    """Test Pinecone connection"""
    print("\n" + "="*60)
    print("2. Testing Pinecone Connection")
    print("="*60)
    
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in .env file")
        return None
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"✅ Pinecone client initialized")
        
        # List indexes
        indexes_list = pc.list_indexes()
        if hasattr(indexes_list, 'indexes'):
            index_names = [idx.name for idx in indexes_list.indexes]
        else:
            index_names = list(indexes_list.names()) if hasattr(indexes_list, 'names') else []
        
        print(f"📋 Available indexes: {index_names}")
        
        if INDEX_NAME not in index_names:
            print(f"⚠️  WARNING: Index '{INDEX_NAME}' not found in available indexes")
            print(f"   Attempting to connect anyway...")
        
        # Connect to index
        index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to index: {INDEX_NAME}")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total vectors: {stats.get('total_vector_count', 'N/A')}")
        print(f"   Dimension: {stats.get('dimension', 'N/A')}")
        print(f"   Index fullness: {stats.get('index_fullness', 'N/A')}")
        
        if stats.get('namespaces'):
            print(f"   Namespaces: {list(stats['namespaces'].keys())}")
        
        return index
        
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify PINECONE_API_KEY in .env file")
        print("2. Check if index exists in Pinecone console")
        print("3. Verify index name: brookstone-faq-ollama")
        return None


def test_retrieval(embeddings, index):
    """Test document retrieval"""
    print("\n" + "="*60)
    print("3. Testing Document Retrieval")
    print("="*60)
    
    test_queries = [
        "What are the flat configurations available?",
        "What amenities does Brookstone offer?",
        "Where is Brookstone located?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        try:
            # Generate embedding
            query_embedding = embeddings.embed_query(query)
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            matches = results.get('matches', [])
            print(f"✅ Retrieved {len(matches)} documents")
            
            if not matches:
                print(f"⚠️  No documents found for this query")
                continue
            
            # Show results
            for j, match in enumerate(matches, 1):
                score = match.get('score', 0)
                metadata = match.get('metadata', {})
                
                # Extract text content
                text = (
                    metadata.get('text') or 
                    metadata.get('content') or 
                    metadata.get('page_content') or
                    metadata.get('chunk') or
                    "No text found"
                )
                
                print(f"\n   Result {j}:")
                print(f"   Score: {score:.4f}")
                print(f"   Text: {text[:100]}...")
                print(f"   Metadata keys: {list(metadata.keys())}")
            
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")


def test_full_pipeline(embeddings, index):
    """Test complete retrieval pipeline with Document conversion"""
    print("\n" + "="*60)
    print("4. Testing Full Pipeline (as in your app)")
    print("="*60)
    
    query = "Tell me about the amenities"
    print(f"Query: {query}")
    
    try:
        # Generate embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Convert to Document objects
        documents = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            
            text_content = (
                metadata.get('text') or 
                metadata.get('content') or 
                metadata.get('page_content') or 
                metadata.get('chunk') or
                ""
            )
            
            doc = Document(
                page_content=text_content,
                metadata={
                    'score': match.get('score', 0),
                    'id': match.get('id', ''),
                    **metadata
                }
            )
            documents.append(doc)
        
        print(f"✅ Converted {len(documents)} results to Document objects")
        
        if documents:
            print(f"\n📄 First document:")
            print(f"   Content length: {len(documents[0].page_content)} chars")
            print(f"   Content preview: {documents[0].page_content[:150]}...")
            print(f"   Score: {documents[0].metadata.get('score', 'N/A'):.4f}")
            print(f"   Metadata keys: {list(documents[0].metadata.keys())}")
            
            # Build context string (as in your app)
            context = "\n\n".join([
                (d.page_content or "") + 
                ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items()))
                for d in documents
            ])
            
            print(f"\n✅ Context string built: {len(context)} characters")
            print(f"   This context will be sent to Gemini for response generation")
        else:
            print(f"⚠️  No documents with text content found")
            print(f"   Check your Pinecone data structure!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BROOKSTONE PINECONE + OLLAMA TEST SUITE")
    print("="*60)
    
    # Test 1: Ollama embeddings
    embeddings = test_ollama_embeddings()
    if not embeddings:
        print("\n❌ Cannot proceed without Ollama embeddings")
        return
    
    # Test 2: Pinecone connection
    index = test_pinecone_connection(embeddings)
    if not index:
        print("\n❌ Cannot proceed without Pinecone connection")
        return
    
    # Test 3: Basic retrieval
    test_retrieval(embeddings, index)
    
    # Test 4: Full pipeline
    test_full_pipeline(embeddings, index)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print("\nIf all tests passed, your app is ready to run!")
    print("If any tests failed, check the troubleshooting tips above.")


if __name__ == "__main__":
    main()