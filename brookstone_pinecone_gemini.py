import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
import json
from datetime import datetime, timedelta
import google.generativeai as genai
from pinecone import Pinecone

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================================================
# ENVIRONMENT VARIABLES
# ================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "brookstone_verify_token_2024")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BROCHURE_URL = os.getenv("BROCHURE_URL", "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf")

WORKVEU_WEBHOOK_URL = os.getenv("WORKVEU_WEBHOOK_URL")
WORKVEU_API_KEY = os.getenv("WORKVEU_API_KEY")

# Media state file
MEDIA_STATE_FILE = os.path.join(os.path.dirname(__file__), "media_state.json")
MEDIA_ID = None
MEDIA_EXPIRY_DAYS = 29

# ================================================
# MEDIA MANAGEMENT
# ================================================
def load_media_state():
    try:
        if os.path.exists(MEDIA_STATE_FILE):
            with open(MEDIA_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading media state: {e}")
    return {}

def save_media_state(state: dict):
    try:
        with open(MEDIA_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"❌ Error saving media state: {e}")

def _find_brochure_file():
    candidates = [
        os.path.join(os.path.dirname(__file__), "static", "brochure", "BROOKSTONE.pdf"),
        os.path.join(os.path.dirname(__file__), "static", "BROCHURE.pdf"),
        os.path.join(os.path.dirname(__file__), "BROOKSTONE.pdf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def upload_brochure_media():
    global MEDIA_ID
    file_path = _find_brochure_file()
    if not file_path:
        logging.error("❌ Brochure file not found")
        return None

    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    data = {"messaging_product": "whatsapp"}
    try:
        with open(file_path, "rb") as fh:
            files = {"file": (os.path.basename(file_path), fh, "application/pdf")}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        if resp.status_code == 200:
            j = resp.json()
            new_media_id = j.get("id")
            if new_media_id:
                MEDIA_ID = new_media_id
                state = {"media_id": MEDIA_ID, "uploaded_at": datetime.utcnow().isoformat()}
                save_media_state(state)
                logging.info(f"✅ Uploaded brochure media, id={MEDIA_ID}")
                return MEDIA_ID
    except Exception as e:
        logging.error(f"❌ Exception uploading media: {e}")
    return None

def ensure_media_up_to_date():
    global MEDIA_ID
    state = load_media_state()
    media_id = state.get("media_id")
    uploaded_at = state.get("uploaded_at")
    need_upload = True
    if media_id and uploaded_at:
        try:
            uploaded_dt = datetime.fromisoformat(uploaded_at)
            if datetime.utcnow() - uploaded_dt < timedelta(days=MEDIA_EXPIRY_DAYS):
                MEDIA_ID = media_id
                need_upload = False
                logging.info(f"ℹ️ Using existing media_id (uploaded {uploaded_at})")
        except Exception:
            need_upload = True

    if need_upload:
        logging.info("ℹ️ Uploading brochure media")
        upload_brochure_media()

try:
    ensure_media_up_to_date()
except Exception as e:
    logging.error(f"❌ Error initializing media: {e}")

# ================================================
# GEMINI SETUP
# ================================================
gemini_model = None
gemini_chat = None

if not GEMINI_API_KEY:
    logging.error("❌ Missing Gemini API key!")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        gemini_chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
        logging.info("✅ Gemini API configured")
    except Exception as e:
        logging.error(f"❌ Error initializing Gemini: {e}")

# ================================================
# PINECONE + OLLAMA SETUP (NO WRAPPER)
# ================================================
INDEX_NAME = "brookstone-faq-ollama"

# Initialize Ollama embeddings
ollama_embeddings = None
try:
    ollama_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    test_embed = ollama_embeddings.embed_query("test")
    logging.info(f"✅ Ollama embeddings initialized (dimension: {len(test_embed)})")
except Exception as e:
    logging.error(f"❌ Error initializing Ollama embeddings: {e}")
    ollama_embeddings = None

# Initialize Pinecone
pc = None
pinecone_index = None

try:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logging.info("✅ Pinecone client created")
    
    pinecone_index = pc.Index(INDEX_NAME)
    logging.info(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    
    try:
        stats = pinecone_index.describe_index_stats()
        logging.info(f"📊 Index stats: Total vectors: {stats.get('total_vector_count', 'N/A')}")
    except Exception as e:
        logging.warning(f"⚠️ Could not get index stats: {e}")
        
except Exception as e:
    logging.error(f"❌ Failed to initialize Pinecone: {e}")
    pc = None
    pinecone_index = None

# ================================================
# DIRECT RETRIEVAL FUNCTION
# ================================================
def retrieve_context(query: str, top_k: int = 5):
    """Retrieve relevant documents from Pinecone using Ollama embeddings"""
    if not ollama_embeddings:
        logging.error("❌ Ollama embeddings not available")
        return []
    
    if not pinecone_index:
        logging.error("❌ Pinecone index not available")
        return []
    
    try:
        query_embedding = ollama_embeddings.embed_query(query)
        logging.info(f"🔍 Generated query embedding (dimension: {len(query_embedding)})")
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        logging.info(f"📚 Retrieved {len(results.get('matches', []))} documents")
        
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
        
        if documents:
            logging.info(f"✅ Successfully converted {len(documents)} results to Document objects")
        else:
            logging.warning("⚠️ No documents with text content retrieved")
        
        return documents
        
    except Exception as e:
        logging.error(f"❌ Error retrieving context: {e}")
        logging.exception("Full traceback:")
        return []

# ================================================
# TRANSLATION FUNCTIONS
# ================================================
def translate_gujarati_to_english(text):
    try:
        if not gemini_model:
            return text
        translation_prompt = f"""
Translate the following Gujarati text to English. Provide only the English translation.

Gujarati text: {text}

English translation:
        """
        response = gemini_model.generate_content(translation_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"❌ Error translating Gujarati to English: {e}")
        return text

def translate_english_to_gujarati(text):
    try:
        if not gemini_model:
            return text
        translation_prompt = f"""
Translate the following English text to Gujarati. Keep it brief and concise.

English text: {text}

Gujarati translation:
        """
        response = gemini_model.generate_content(translation_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"❌ Error translating English to Gujarati: {e}")
        return text

# ================================================
# CONVERSATION STATE
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {
            "chat_history": [], 
            "language": "english",
            "waiting_for": None,
            "last_context_topics": [],
            "user_interests": [],
            "last_follow_up": None,
            "follow_up_context": None,
            "is_first_message": True,
            "conversation_summary": "",
            "user_preferences": {}
        }
    else:
        for key in ["waiting_for", "last_context_topics", "user_interests", "last_follow_up", 
                    "follow_up_context", "is_first_message", "conversation_summary", "user_preferences"]:
            if key not in CONV_STATE[from_phone]:
                CONV_STATE[from_phone][key] = None if key in ["waiting_for", "last_follow_up", "follow_up_context"] else (
                    [] if key in ["last_context_topics", "user_interests", "chat_history"] else (
                    True if key == "is_first_message" else ("" if key == "conversation_summary" else {})
                    )
                )

def update_conversation_memory_with_gemini(state, user_message, bot_response):
    try:
        if not gemini_model or len(state["chat_history"]) % 4 != 0:
            return
            
        recent_history = state["chat_history"][-8:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        memory_prompt = f"""
Analyze this conversation and provide JSON:
{history_text}

{{
    "summary": "brief summary",
    "preferences": {{
        "budget_range": "unknown",
        "preferred_bhk": "unknown",
        "key_interests": [],
        "visit_intent": "unknown"
    }}
}}
        """
        
        response = gemini_model.generate_content(memory_prompt)
        try:
            memory_data = json.loads(response.text.strip())
            state["conversation_summary"] = memory_data.get("summary", "")
            state["user_preferences"].update(memory_data.get("preferences", {}))
        except:
            state["conversation_summary"] = response.text.strip()[:200]
    except Exception as e:
        logging.error(f"❌ Error updating memory: {e}")

def detect_location_request_with_gemini(message_text):
    try:
        if not gemini_model:
            return False
        
        property_keywords = ["bhk", "flat", "apartment", "price", "cost", "interested", "ફ્લેટ", "ઘર", "કિંમત"]
        if any(k in message_text.lower() for k in property_keywords):
            return False
        
        prompt = f"""
Is this a location/address request? Answer only YES or NO.
Message: "{message_text}"
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip().upper() == "YES"
    except Exception as e:
        logging.error(f"❌ Error in location detection: {e}")
        return False

def analyze_user_interests_with_gemini(message_text, state):
    try:
        if not gemini_model:
            return []
        
        prompt = f"""
Analyze interests: pricing, size, amenities, location, availability, visit, brochure
Message: "{message_text}"
Return comma-separated list only.
        """
        response = gemini_model.generate_content(prompt)
        interests = [i.strip() for i in response.text.strip().split(",") if i.strip()]
        state["user_interests"].extend(interests)
        state["user_interests"] = list(set(state["user_interests"][-5:]))
        return interests
    except Exception as e:
        logging.error(f"❌ Error analyzing interests: {e}")
        return []

# ================================================
# WHATSAPP FUNCTIONS
# ================================================
def send_whatsapp_text(to_phone, message):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to_phone, "type": "text", "text": {"body": message}}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Message sent to {to_phone}")
        else:
            logging.error(f"❌ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error: {e}")

def send_whatsapp_location(to_phone):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "location",
        "location": {
            "latitude": "23.0433468",
            "longitude": "72.4594457",
            "name": "Brookstone",
            "address": "Brookstone, Vaikunth Bungalows, Beside DPS Bopal Rd, next to A. Shridhar Oxygen Park, Bopal, Shilaj, Ahmedabad, Gujarat 380058"
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Location sent")
        else:
            logging.error(f"❌ Failed: {response.text}")
    except Exception as e:
        logging.error(f"❌ Error: {e}")

def send_whatsapp_document(to_phone, caption="Here is your Brookstone Brochure 📄"):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    
    if MEDIA_ID:
        payload = {"messaging_product": "whatsapp", "to": to_phone, "type": "document",
                  "document": {"id": MEDIA_ID, "caption": caption, "filename": "Brookstone_Brochure.pdf"}}
    else:
        payload = {"messaging_product": "whatsapp", "to": to_phone, "type": "document",
                  "document": {"link": BROCHURE_URL, "caption": caption, "filename": "Brookstone_Brochure.pdf"}}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Document sent")
        else:
            logging.error(f"❌ Failed: {response.text}")
    except Exception as e:
        logging.error(f"❌ Error: {e}")

def mark_message_as_read(message_id):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Error marking read: {e}")

# ================================================
# WORKVEU CRM
# ================================================
def push_to_workveu(name, wa_id, message_text, direction="inbound"):
    if not WORKVEU_WEBHOOK_URL or not WORKVEU_API_KEY:
        return
    payload = {
        "api_key": WORKVEU_API_KEY,
        "contacts": [{
            "profile": {"name": name or "Unknown"},
            "wa_id": wa_id,
            "remarks": f"[{direction.upper()}] {message_text}"
        }]
    }
    try:
        response = requests.post(WORKVEU_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info(f"✅ WorkVEU synced ({direction})")
    except Exception as e:
        logging.error(f"❌ WorkVEU error: {e}")

# ================================================
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    ensure_conversation_state(from_phone)
    state = CONV_STATE[from_phone]
    
    # Language detection
    current_msg_has_gujarati = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    if current_msg_has_gujarati:
        state["language"] = "gujarati"
    else:
        english_chars = sum(1 for c in message_text if c.isalpha() and ord(c) < 128)
        total_chars = len([c for c in message_text if c.isalpha()])
        if total_chars > 0 and (english_chars / total_chars) > 0.7:
            state["language"] = "english"
        else:
            recent_gujarati = any(any("\u0A80" <= c <= "\u0AFF" for c in msg.get("content", ""))
                                for msg in state["chat_history"][-3:])
            state["language"] = "gujarati" if recent_gujarati else "english"
    
    logging.info(f"🌐 Language: {state['language']} | Message: {message_text[:30]}...")
    
    state["chat_history"].append({"role": "user", "content": message_text})
    push_to_workveu(name=None, wa_id=from_phone, message_text=message_text, direction="inbound")

    # First message welcome
    if state.get("is_first_message", True):
        state["is_first_message"] = False
        welcome = "Hello! Welcome to Brookstone. How could I assist you today? 🏠✨"
        if state["language"] == "gujarati":
            welcome = translate_english_to_gujarati(welcome)
        send_whatsapp_text(from_phone, welcome)
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=welcome, direction="outbound")
        return

    # Location request check
    if detect_location_request_with_gemini(message_text):
        logging.info(f"🗺️ Location request detected")
        send_whatsapp_location(from_phone)
        return

    # Analyze interests
    current_interests = analyze_user_interests_with_gemini(message_text, state)
    logging.info(f"📱 Interests: {current_interests}")

    message_lower = message_text.lower().strip()
    
    # Handle brochure confirmation
    if state.get("waiting_for") == "brochure_confirmation":
        if any(w in message_lower for w in ["yes", "yeah", "sure", "send", "ok", "હા", "જી"]):
            state["waiting_for"] = None
            send_whatsapp_document(from_phone)
            brochure_text = "📄 Here's your Brookstone brochure! Any questions? ✨😊"
            if state["language"] == "gujarati":
                brochure_text = translate_english_to_gujarati(brochure_text)
            send_whatsapp_text(from_phone, brochure_text)
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=brochure_text, direction="outbound")
            return
        elif any(w in message_lower for w in ["no", "later", "નહીં"]):
            state["waiting_for"] = None
            later_text = "Sure! Let me know if you need anything. 🏠😊"
            if state["language"] == "gujarati":
                later_text = translate_english_to_gujarati(later_text)
            send_whatsapp_text(from_phone, later_text)
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=later_text, direction="outbound")
            return

    if not pinecone_index or not ollama_embeddings:
        error = "Please contact 8238477697 or 9974812701 for info."
        if state["language"] == "gujarati":
            error = translate_english_to_gujarati(error)
        send_whatsapp_text(from_phone, error)
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=error, direction="outbound")
        return

    try:
        # Translate query if needed
        search_query = message_text
        if state["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)
            logging.info(f"🔄 Translated: {search_query}")

        # ⭐ DIRECT RETRIEVAL
        docs = retrieve_context(search_query, top_k=5)
        logging.info(f"📚 Retrieved {len(docs)} documents")

        if not docs:
            context = ""
        else:
            context = "\n\n".join([
                (d.page_content or "") + 
                ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items()))
                for d in docs
            ])

        state["last_context_topics"] = [d.metadata.get("topic", "") for d in docs if d.metadata.get("topic")]

        # Build system prompt
        user_context = f"User interests: {', '.join(state['user_interests'])}" if state['user_interests'] else "New"
        conversation_context = f"\nSUMMARY: {state['conversation_summary']}" if state.get("conversation_summary") else ""
        
        user_prefs = state.get("user_preferences", {})
        prefs_context = ""
        if user_prefs:
            items = []
            for key in ["budget_range", "preferred_bhk", "visit_intent"]:
                val = user_prefs.get(key, "unknown")
                if val != "unknown":
                    items.append(f"{key}: {val}")
            if items:
                prefs_context = f"\nPREFS: {' | '.join(items)}"
        
        follow_up_mem = ""
        if state.get("last_follow_up"):
            follow_up_mem = f"\nLAST Q: '{state['last_follow_up']}'"
        
        lang_inst = ""
        if state["language"] == "gujarati":
            lang_inst = "User speaks Gujarati. Respond in ENGLISH (keep VERY SHORT), will auto-translate."

        system_prompt = f"""
You are Brookstone's friendly assistant. Be conversational, natural, convincing.

{lang_inst}

CORE:
- EXTREMELY CONCISE - Max 1-2 sentences
- Use 2-3 emojis
- NO long explanations
- CONVINCE user friendly
- Be NATURAL - don't repeat phrases

MEMORY: {follow_up_mem}{conversation_context}{prefs_context}

FLAT MENTIONS:
- Only mention "3&4BHK" when user asks about types/configurations
- Don't force into every response

BROCHURE:
- Detect when user wants details/plans
- Offer: "I'll send detailed brochure!"

SPECIAL:
1. TIMINGS: "Site office: *10:30 AM - 7:00 PM* daily �"
2. SITE VISIT: "Contact *Mr. Nilesh at 7600612701* 📞✨"
3. GENERAL: "Contact 8238477697 or 9974812701 📱😊"
4. PRICING: "Contact 8238477697/9974812701 for pricing 💰📞"

USER: {user_context}

Context:
{context}

Query: {search_query}

Keep VERY SHORT - Max 1-2 sentences + 1 follow-up!
Assistant:
        """.strip()

        if not gemini_chat:
            error = "AI unavailable. Contact 8238477697/9974812701"
            if state["language"] == "gujarati":
                error = translate_english_to_gujarati(error)
            send_whatsapp_text(from_phone, error)
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=error, direction="outbound")
            return

        response = gemini_chat.invoke(system_prompt).content.strip()
        logging.info(f"🧠 Response: {response}")

        final_response = response
        if state["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)

        send_whatsapp_text(from_phone, final_response)
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=final_response, direction="outbound")

        # Extract follow-up
        for sentence in final_response.split('.'):
            if '?' in sentence:
                state["last_follow_up"] = sentence.strip()
                break

        # Check response for states
        resp_lower = response.lower()
        if ("layout" in resp_lower or "details" in resp_lower) and "site visit" in resp_lower:
            state["waiting_for"] = "clarification_needed"
        elif "would you like" in resp_lower and "brochure" in resp_lower:
            state["waiting_for"] = "brochure_confirmation"
        
        if re.search(r"\bi'll send.*brochure\b", resp_lower) and state.get("waiting_for") != "brochure_confirmation":
            send_whatsapp_document(from_phone)

        state["chat_history"].append({"role": "assistant", "content": final_response})
        update_conversation_memory_with_gemini(state, message_text, final_response)

    except Exception as e:
        logging.error(f"❌ RAG error: {e}")
        logging.exception("Traceback:")
        error = "Technical issue. Contact 8238477697/9974812701"
        if state["language"] == "gujarati":
            error = translate_english_to_gujarati(error)
        send_whatsapp_text(from_phone, error)
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=error, direction="outbound")

# ================================================
# ROUTES
# ================================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("✅ WEBHOOK VERIFIED")
        return challenge, 200
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                for message in change.get("value", {}).get("messages", []):
                    from_phone = message.get("from")
                    message_id = message.get("id")
                    msg_type = message.get("type")

                    text = ""
                    if msg_type == "text":
                        text = message.get("text", {}).get("body", "")
                    elif msg_type == "button":
                        text = message.get("button", {}).get("text", "")
                    elif msg_type == "interactive":
                        interactive = message.get("interactive", {})
                        if "button_reply" in interactive:
                            text = interactive["button_reply"].get("title", "")
                        elif "list_reply" in interactive:
                            text = interactive["list_reply"].get("title", "")

                    if not text:
                        continue

                    mark_message_as_read(message_id)
                    process_incoming_message(from_phone, text, message_id)

    except Exception as e:
        logging.exception("❌ Error processing webhook")

    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "whatsapp_configured": bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "gemini_configured": bool(GEMINI_API_KEY and gemini_model and gemini_chat),
        "pinecone_configured": bool(PINECONE_API_KEY and pinecone_index),
        "ollama_configured": bool(ollama_embeddings),
        "workveu_configured": bool(WORKVEU_WEBHOOK_URL and WORKVEU_API_KEY),
        "retrieval_ready": bool(pinecone_index and ollama_embeddings)
    }), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Brookstone WhatsApp RAG Bot is running!",
        "brochure_url": BROCHURE_URL,
        "endpoints": {"webhook": "/webhook", "health": "/health"}
    }), 200

# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp Bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)