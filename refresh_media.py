#!/usr/bin/env python3
"""
Standalone script to refresh WhatsApp media ID for Brookstone brochure.
This script is designed to be run by Windows Task Scheduler every 29 days.

Usage:
    python refresh_media.py

Requirements:
    - .env file with WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID
    - BROOKSTONE.pdf in static/brochure/ or static/ folder
"""

import os
import sys
import logging
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("refresh_media.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
MEDIA_STATE_FILE = os.path.join(os.path.dirname(__file__), "media_state.json")

def load_media_state():
    """Load media state from file."""
    try:
        if os.path.exists(MEDIA_STATE_FILE):
            with open(MEDIA_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading media state: {e}")
    return {}

def save_media_state(state: dict):
    """Save media state to file."""
    try:
        with open(MEDIA_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logging.info(f"💾 Media state saved to {MEDIA_STATE_FILE}")
    except Exception as e:
        logging.error(f"❌ Error saving media state: {e}")

def find_brochure_file():
    """Find the brochure PDF file in common locations."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "static", "brochure", "BROOKSTONE.pdf"),
        os.path.join(os.path.dirname(__file__), "static", "brochure", "BROOKSTONE .pdf"),  # with space
        os.path.join(os.path.dirname(__file__), "static", "BROOKSTONE.pdf"),
        os.path.join(os.path.dirname(__file__), "static", "BROOKSTONE .pdf"),  # with space
        os.path.join(os.path.dirname(__file__), "BROOKSTONE.pdf"),
        os.path.join(os.path.dirname(__file__), "BROOKSTONE .pdf"),  # with space
        os.path.join(os.path.dirname(__file__), "Brookstone.pdf"),  # different case
        os.path.join(os.path.dirname(__file__), "brookstone.pdf"),  # lowercase
    ]
    
    for path in candidates:
        if os.path.exists(path):
            logging.info(f"📄 Found brochure file: {path}")
            return path
    
    logging.error("❌ Brochure file not found in any of these locations:")
    for path in candidates:
        logging.error(f"   - {path}")
    return None

def upload_brochure_media():
    """Upload brochure PDF to WhatsApp Cloud API and return media ID."""
    
    # Validation
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logging.error("❌ Missing WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID in environment variables")
        return None
    
    # Find brochure file
    file_path = find_brochure_file()
    if not file_path:
        return None
    
    # Upload to WhatsApp Cloud API
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    data = {"messaging_product": "whatsapp"}
    
    try:
        with open(file_path, "rb") as file_handle:
            files = {"file": (os.path.basename(file_path), file_handle, "application/pdf")}
            logging.info(f"🚀 Uploading {os.path.basename(file_path)} to WhatsApp Cloud API...")
            
            response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            media_id = result.get("id")
            
            if media_id:
                # Save media state
                state = {
                    "media_id": media_id,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "file_name": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path)
                }
                save_media_state(state)
                
                logging.info(f"✅ Successfully uploaded brochure!")
                logging.info(f"📋 Media ID: {media_id}")
                logging.info(f"⏰ Uploaded at: {state['uploaded_at']}")
                
                return media_id
            else:
                logging.error(f"❌ Upload succeeded but no media ID returned: {response.text}")
        else:
            logging.error(f"❌ Upload failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        logging.error(f"❌ Exception during upload: {e}")
    
    return None

def check_media_status():
    """Check current media status and age."""
    state = load_media_state()
    
    if not state:
        logging.info("ℹ️ No existing media state found")
        return False
    
    media_id = state.get("media_id")
    uploaded_at = state.get("uploaded_at")
    
    if not media_id or not uploaded_at:
        logging.info("ℹ️ Incomplete media state found")
        return False
    
    try:
        uploaded_dt = datetime.fromisoformat(uploaded_at)
        age_days = (datetime.utcnow() - uploaded_dt).days
        
        logging.info(f"📊 Current media status:")
        logging.info(f"   Media ID: {media_id}")
        logging.info(f"   Uploaded: {uploaded_at}")
        logging.info(f"   Age: {age_days} days")
        
        if age_days >= 29:
            logging.info("⚠️ Media is 29+ days old and needs refresh")
            return False
        else:
            logging.info(f"✅ Media is still fresh ({29 - age_days} days remaining)")
            return True
    
    except Exception as e:
        logging.error(f"❌ Error parsing upload date: {e}")
        return False

def main():
    """Main function - refresh media if needed."""
    logging.info("🚀 Starting WhatsApp media refresh check...")
    logging.info(f"📁 Working directory: {os.getcwd()}")
    logging.info(f"📄 Script location: {__file__}")
    
    # Check if refresh is needed
    if check_media_status():
        logging.info("✅ Media is still fresh, no refresh needed")
        return True
    
    # Upload new media
    logging.info("🔄 Refreshing media...")
    media_id = upload_brochure_media()
    
    if media_id:
        logging.info("🎉 Media refresh completed successfully!")
        return True
    else:
        logging.error("❌ Media refresh failed!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        logging.info(f"🏁 Script completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logging.error(f"💥 Unexpected error: {e}")
        sys.exit(1)