import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
import json
from datetime import datetime, timedelta
import google.generativeai as genai

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BROCHURE_URL = os.getenv("BROCHURE_URL", "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf")

# WorkVEU CRM Integration
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
    """Load media state from file."""
    try:
        if os.path.exists(MEDIA_STATE_FILE):
            with open(MEDIA_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading media state: {e}")
    return {}

def save_media_state(state):
    """Save media state to file."""
    try:
        with open(MEDIA_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"❌ Error saving media state: {e}")

def _find_brochure_file():
    """Find brochure PDF file."""
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
    """Upload brochure PDF to WhatsApp Cloud."""
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
                state = {"media_id": MEDIA_ID, "uploaded_at": datetime.now(datetime.UTC).isoformat()}
                save_media_state(state)
                logging.info(f"✅ Uploaded brochure media, id={MEDIA_ID}")
                return MEDIA_ID
            else:
                logging.error(f"❌ No media id returned: {resp.text}")
        else:
            logging.error(f"❌ Failed to upload: {resp.status_code} - {resp.text}")
    except Exception as e:
        logging.error(f"❌ Exception uploading media: {e}")
    return None

def ensure_media_up_to_date():
    """Ensure media_id is valid and not expired."""
    global MEDIA_ID
    state = load_media_state()
    media_id = state.get("media_id")
    uploaded_at = state.get("uploaded_at")
    need_upload = True
    
    if media_id and uploaded_at:
        try:
            uploaded_dt = datetime.fromisoformat(uploaded_at)
            if datetime.now(datetime.UTC) - uploaded_dt < timedelta(days=MEDIA_EXPIRY_DAYS):
                MEDIA_ID = media_id
                need_upload = False
                logging.info(f"ℹ️ Using existing media_id")
        except Exception:
            need_upload = True

    if need_upload:
        logging.info("ℹ️ Uploading brochure media")
        upload_brochure_media()

# Initialize media
try:
    ensure_media_up_to_date()
except Exception as e:
    logging.error(f"❌ Error initializing media: {e}")

# ================================================
# AI MODELS INITIALIZATION
# ================================================
gemini_model = None
gemini_chat = None
openai_embeddings = None

# Initialize Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        gemini_chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
        logging.info("✅ Gemini configured")
    except Exception as e:
        logging.error(f"❌ Gemini initialization error: {e}")
else:
    logging.error("❌ Missing GEMINI_API_KEY")

# Initialize OpenAI embeddings
if OPENAI_API_KEY:
    try:
        openai_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY
        )
        logging.info("✅ OpenAI embeddings configured")
    except Exception as e:
        logging.error(f"❌ OpenAI embeddings error: {e}")
else:
    logging.error("❌ Missing OPENAI_API_KEY")

# ================================================
# PINECONE SETUP
# ================================================
INDEX_NAME = "brookstone-faq-json"
vectorstore = None
retriever = None

def load_vectorstore():
    """Load Pinecone vectorstore."""
    if not openai_embeddings or not PINECONE_API_KEY:
        logging.error("❌ Missing embeddings or Pinecone API key")
        return None
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        vectorstore = PineconeVectorStore(index, openai_embeddings.embed_query, "text")
        return vectorstore
    except Exception as e:
        logging.error(f"❌ Pinecone error: {e}")
        return None

try:
    vectorstore = load_vectorstore()
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logging.info("✅ Pinecone loaded")
    else:
        logging.error("❌ Vectorstore failed")
except Exception as e:
    logging.error(f"❌ Pinecone loading error: {e}")

# ================================================
# TRANSLATION FUNCTIONS
# ================================================
def translate_gujarati_to_english(text):
    """Translate Gujarati to English using Gemini."""
    try:
        if not gemini_model:
            return text
            
        prompt = f"Translate this Gujarati text to English. Only provide the translation:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        translated = response.text.strip()
        logging.info(f"🔄 Translated: {translated[:50]}...")
        return translated
    except Exception as e:
        logging.error(f"❌ Translation error: {e}")
        return text

def translate_english_to_gujarati(text):
    """Translate English to Gujarati using Gemini."""
    try:
        if not gemini_model:
            return text
            
        prompt = f"Translate this English text to Gujarati. Keep it brief. Only provide the translation:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        translated = response.text.strip()
        logging.info(f"🔄 Translated: {translated[:50]}...")
        return translated
    except Exception as e:
        logging.error(f"❌ Translation error: {e}")
        return text

# ================================================
# AREA INFORMATION
# ================================================
AREA_INFO = {
    "3bhk": {"super_buildup": "2650 sqft", "display_name": "3BHK"},
    "4bhk": {"super_buildup": "3850 sqft", "display_name": "4BHK"},
    "3bhk_tower_duplex": {"super_buildup": "5300 sqft + 700 sqft carpet terrace", "display_name": "3BHK Tower Duplex"},
    "4bhk_tower_duplex": {"super_buildup": "7700 sqft + 1000 sqft carpet terrace", "display_name": "4BHK Tower Duplex"},
    "3bhk_tower_simplex": {"super_buildup": "2650 sqft + 700 sqft carpet terrace", "display_name": "3BHK Tower Simplex"},
    "4bhk_tower_simplex": {"super_buildup": "3850 sqft + 1000 sqft carpet terrace", "display_name": "4BHK Tower Simplex"}
}

def get_area_information(query):
    """Get area info from database."""
    query_lower = query.lower()
    results = []
    
    if "tower duplex" in query_lower:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk_tower_duplex']['display_name']}: {AREA_INFO['3bhk_tower_duplex']['super_buildup']}")
        elif "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk_tower_duplex']['display_name']}: {AREA_INFO['4bhk_tower_duplex']['super_buildup']}")
        else:
            results.append(f"{AREA_INFO['3bhk_tower_duplex']['display_name']}: {AREA_INFO['3bhk_tower_duplex']['super_buildup']}")
            results.append(f"{AREA_INFO['4bhk_tower_duplex']['display_name']}: {AREA_INFO['4bhk_tower_duplex']['super_buildup']}")
    elif "tower simplex" in query_lower:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk_tower_simplex']['display_name']}: {AREA_INFO['3bhk_tower_simplex']['super_buildup']}")
        elif "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk_tower_simplex']['display_name']}: {AREA_INFO['4bhk_tower_simplex']['super_buildup']}")
        else:
            results.append(f"{AREA_INFO['3bhk_tower_simplex']['display_name']}: {AREA_INFO['3bhk_tower_simplex']['super_buildup']}")
            results.append(f"{AREA_INFO['4bhk_tower_simplex']['display_name']}: {AREA_INFO['4bhk_tower_simplex']['super_buildup']}")
    else:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk']['display_name']}: {AREA_INFO['3bhk']['super_buildup']}")
        if "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk']['display_name']}: {AREA_INFO['4bhk']['super_buildup']}")
        
        if not results and any(term in query_lower for term in ["what are", "all", "available", "sizes", "options"]):
            if not any(specific in query_lower for specific in ["5bhk", "penthouse", "villa", "studio"]):
                results = [
                    f"{AREA_INFO['3bhk']['display_name']}: {AREA_INFO['3bhk']['super_buildup']}",
                    f"{AREA_INFO['4bhk']['display_name']}: {AREA_INFO['4bhk']['super_buildup']}"
                ]
    
    return results

# ================================================
# CONVERSATION STATE
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    """Ensure conversation state exists."""
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
        # Ensure all fields exist
        defaults = {
            "waiting_for": None,
            "last_context_topics": [],
            "user_interests": [],
            "last_follow_up": None,
            "follow_up_context": None,
            "is_first_message": True,
            "conversation_summary": "",
            "user_preferences": {}
        }
        for key, value in defaults.items():
            if key not in CONV_STATE[from_phone]:
                CONV_STATE[from_phone][key] = value

def detect_location_request_with_gemini(message_text):
    """Detect location requests using Gemini."""
    try:
        if not gemini_model:
            return False
        
        property_keywords = ["bhk", "flat", "apartment", "house", "property", "unit", "bedroom", "price"]
        if any(kw in message_text.lower() for kw in property_keywords):
            return False
        
        prompt = f"""Is this message asking for location/address/directions? Answer only YES or NO.
        
Message: "{message_text}"

Only YES if asking for: location, address, directions, map, where is it, how to reach
Only NO for: property inquiries, pricing, features, availability"""
        
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().upper()
        return result == "YES"
    except Exception as e:
        logging.error(f"❌ Location detection error: {e}")
        return False

# ================================================
# WHATSAPP FUNCTIONS
# ================================================
def send_whatsapp_text(to_phone, message):
    """Send text message via WhatsApp."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": message}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Message sent to {to_phone}")
        else:
            logging.error(f"❌ Send failed: {response.status_code}")
    except Exception as e:
        logging.error(f"❌ Send error: {e}")

def send_whatsapp_location(to_phone):
    """Send location via WhatsApp."""
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
            "address": "Brookstone, Vaikunth Bungalows, Beside DPS Bopal Rd, Bopal, Ahmedabad, Gujarat 380058"
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Location sent")
        else:
            logging.error(f"❌ Location send failed")
    except Exception as e:
        logging.error(f"❌ Location error: {e}")

def send_whatsapp_document(to_phone, caption="Here is your Brookstone Brochure 📄"):
    """Send document via WhatsApp."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    if MEDIA_ID:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "document",
            "document": {"id": MEDIA_ID, "caption": caption, "filename": "Brookstone_Brochure.pdf"}
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "document",
            "document": {"link": BROCHURE_URL, "caption": caption, "filename": "Brookstone_Brochure.pdf"}
        }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Document sent")
        else:
            logging.error(f"❌ Document send failed")
    except Exception as e:
        logging.error(f"❌ Document error: {e}")

def mark_message_as_read(message_id):
    """Mark message as read."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Read error: {e}")

# ================================================
# WORKVEU CRM
# ================================================
def push_to_workveu(name, wa_id, message_text, direction="inbound"):
    """Push to WorkVEU CRM."""
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
            logging.info(f"✅ WorkVEU synced")
    except Exception as e:
        logging.error(f"❌ WorkVEU error: {e}")

# ================================================
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    """Process incoming WhatsApp message."""
    ensure_conversation_state(from_phone)
    state = CONV_STATE[from_phone]
    
    # Detect language
    has_gujarati = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    state["language"] = "gujarati" if has_gujarati else "english"
    
    state["chat_history"].append({"role": "user", "content": message_text})
    push_to_workveu(None, from_phone, message_text, "inbound")

    # Welcome message
    if state.get("is_first_message", True):
        state["is_first_message"] = False
        welcome = "Hello! Welcome to Brookstone. How can I assist you today? 🏠✨"
        if state["language"] == "gujarati":
            welcome = translate_english_to_gujarati(welcome)
        send_whatsapp_text(from_phone, welcome)
        push_to_workveu("Brookstone Bot", from_phone, welcome, "outbound")
        return

    # Location request
    if detect_location_request_with_gemini(message_text):
        send_whatsapp_location(from_phone)
        return

    # Handle confirmations
    msg_lower = message_text.lower().strip()
    
    if state.get("waiting_for") == "brochure_confirmation":
        if any(w in msg_lower for w in ["yes", "sure", "send", "ok", "હા"]):
            state["waiting_for"] = None
            send_whatsapp_document(from_phone)
            msg = "📄 Here's your brochure! Any questions? ✨"
            if state["language"] == "gujarati":
                msg = translate_english_to_gujarati(msg)
            send_whatsapp_text(from_phone, msg)
            push_to_workveu("Bot", from_phone, msg, "outbound")
            return
        elif any(w in msg_lower for w in ["no", "later", "ના"]):
            state["waiting_for"] = None
            msg = "Sure! Let me know if you need anything else. 😊"
            if state["language"] == "gujarati":
                msg = translate_english_to_gujarati(msg)
            send_whatsapp_text(from_phone, msg)
            push_to_workveu("Bot", from_phone, msg, "outbound")
            return

    if not retriever:
        error = "Please contact 8238477697 or 9974812701 for assistance."
        if state["language"] == "gujarati":
            error = translate_english_to_gujarati(error)
        send_whatsapp_text(from_phone, error)
        push_to_workveu("Bot", from_phone, error, "outbound")
        return

    try:
        # Translate if needed
        search_query = message_text
        if state["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)

        # Search Pinecone
        docs = retriever.invoke(search_query)
        context = "\n\n".join([d.page_content for d in docs])

        # Build prompt
        system_prompt = f"""You are a friendly Brookstone real estate assistant.

INSTRUCTIONS:
- Be concise (1-2 sentences max)
- Use 2-3 emojis
- Answer from context below
- Don't invent details
- Be natural and convincing

AREA TERMINOLOGY:
- NEVER say "carpet area"
- ALWAYS say "Super Build-up area" or "SBU"

CONTACT INFO:
- Timings: 10:30 AM - 7:00 PM daily
- Site visit: Mr. Nilesh at 7600612701
- General: 8238477697 or 9974812701

Context:
{context}

User Question: {search_query}

Response (brief and friendly):"""

        if not gemini_chat:
            raise Exception("Gemini not available")

        response = gemini_chat.invoke(system_prompt).content.strip()
        
        # Replace carpet area globally
        response = re.sub(r'\bcarpet\s+area\b', 'Super Build-up area', response, flags=re.IGNORECASE)

        # Translate if needed
        final_response = response
        if state["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)

        send_whatsapp_text(from_phone, final_response)
        push_to_workveu("Bot", from_phone, final_response, "outbound")

        # Check for brochure offer
        if "brochure" in response.lower() and "would you like" in response.lower():
            state["waiting_for"] = "brochure_confirmation"

        state["chat_history"].append({"role": "assistant", "content": final_response})

    except Exception as e:
        logging.error(f"❌ Processing error: {e}")
        error = "Technical issue. Contact 8238477697 / 9974812701."
        if state["language"] == "gujarati":
            error = translate_english_to_gujarati(error)
        send_whatsapp_text(from_phone, error)
        push_to_workveu("Bot", from_phone, error, "outbound")

# ================================================
# WEBHOOK ROUTES
# ================================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verify webhook."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("✅ Webhook verified")
        return challenge, 200
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming webhooks."""
    data = request.get_json()
    
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                messages = change.get("value", {}).get("messages", [])
                
                for message in messages:
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

                    if text:
                        mark_message_as_read(message_id)
                        process_incoming_message(from_phone, text, message_id)

    except Exception as e:
        logging.exception("❌ Webhook error")

    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "healthy",
        "whatsapp": bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "gemini": bool(gemini_chat),
        "pinecone": bool(retriever),
        "workveu": bool(WORKVEU_WEBHOOK_URL)
    }), 200

@app.route("/", methods=["GET"])
def home():
    """Home route."""
    return jsonify({
        "message": "Brookstone WhatsApp Bot",
        "endpoints": {"/webhook": "Webhook", "/health": "Health"}
    }), 200

# ================================================
# RUN
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)