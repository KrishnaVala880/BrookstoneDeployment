import os
import logging
import time
from datetime import datetime, timedelta
from typing import List

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from openai import OpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------
# Load environment
# ---------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# FIX 1: LangChain expects GOOGLE_API_KEY, not GEMINI_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MEDIA_EXPIRY_DAYS = 29

# ---------------------------------------------
# Logging
# ---------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------
# Gemini Setup
# ---------------------------------------------
try:
    gemini_chat = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,  # FIX: Use google_api_key parameter
        temperature=0.7
    )

    gemini_embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY  # FIX: Use google_api_key parameter
    )

    logging.info("✅ Gemini API configured for chat and translations")

except Exception as e:
    logging.error(f"❌ Gemini initialization error: {e}")
    gemini_chat = None
    gemini_embedder = None

# ---------------------------------------------
# OpenAI client (new embeddings)
# FIX 2: Handle potential proxy issues
# ---------------------------------------------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("✅ OpenAI client initialized")
except TypeError as e:
    # If there's a proxies issue, try without additional params
    logging.warning(f"⚠️ OpenAI client init issue, retrying: {e}")
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------
# Custom Embedding Function (1536 dims)
# ---------------------------------------------
def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Get embedding from text-embedding-3-small with retry logic."""
    if not text or not text.strip():
        raise ValueError("Empty text passed for embedding.")

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return resp.data[0].embedding

        except Exception as e:
            logging.warning(f"⚠️ Embedding retry {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

# ---------------------------------------------
# Wrapper for LangChain VectorStore
# ---------------------------------------------
class CustomEmbedding:
    """Used by LangChain Pinecone vectorstore."""
    def embed_query(self, text: str):
        return get_embedding(text)

    def embed_documents(self, texts: List[str]):
        return [get_embedding(t) for t in texts]


# ---------------------------------------------
# Pinecone Setup
# ---------------------------------------------
logging.info("🔄 Initializing Pinecone...")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        logging.info("⛔ Index not found. Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    logging.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")

except Exception as e:
    logging.error(f"❌ Pinecone initialization failed: {e}")
    index = None

# ---------------------------------------------
# Create VectorStore (with custom embeddings)
# ---------------------------------------------
try:
    openai_embeddings = CustomEmbedding()

    vectorstore = LangchainPinecone(
        index,
        openai_embeddings.embed_query,
        "text"
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logging.info("📚 Pinecone retriever ready.")

except Exception as e:
    logging.error(f"❌ Failed to initialize vectorstore: {e}")
    vectorstore = None
    retriever = None

# ---------------------------------------------
# WhatsApp Bot Logic
# ---------------------------------------------
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        incoming = request.json
        user_msg = incoming.get("message", "")

        if retriever:
            docs = retriever.get_relevant_documents(user_msg)
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = "No context available."

        prompt = f"""
You are a helpful bot. Use the context below to answer.

Context:
{context}

User message:
{user_msg}
"""

        response = gemini_chat.invoke(prompt)
        answer = response.content

        return jsonify({"reply": answer})

    except Exception as e:
        logging.error(f"❌ Error in webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Brookstone WhatsApp Bot is running."


# ---------------------------------------------
# Run Server
# ---------------------------------------------
if __name__ == "__main__":
    logging.info("🚀 Starting Brookstone WhatsApp Bot on port 5000")
    app.run(host="0.0.0.0", port=5000)