from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import os
import openai
from typing import Optional
from datetime import datetime

# --- Init ---
app = FastAPI()

# Supabase setup
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Models ---
class CharacterIn(BaseModel):
    name: str
    persona: dict  # persona as JSON

class ChatIn(BaseModel):
    character_id: str
    message: str
    user_id: str  # required now for memories

class MemoryIn(BaseModel):
    user_id: str
    character_id: str
    message: str
    reply: Optional[str] = None

# --- Routes ---

@app.get("/")
def root():
    return {"message": "AI Girlfriend Backend is running 🚀"}

# Create a character
@app.post("/characters")
def create_character(payload: CharacterIn):
    try:
        result = supabase.table("characters").insert({
            "name": payload.name,
            "persona": payload.persona,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        if result.data:
            return {"status": "success", "character": result.data[0]}
        else:
            raise HTTPException(status_code=400, detail="Insert failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get all characters
@app.get("/characters")
def get_characters():
    try:
        result = supabase.table("characters").select("*").execute()
        return {"characters": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat with a character
@app.post("/chat")
def chat(payload: ChatIn):
    try:
        # Fetch character
        char_result = supabase.table("characters").select("*").eq("id", payload.character_id).execute()
        if not char_result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        character = char_result.data[0]

        persona = character.get("persona", {})
        persona_text = f"This is {character['name']}, {persona.get('description', 'a virtual companion')}."

        # Call OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": persona_text},
                {"role": "user", "content": payload.message}
            ]
        )

        reply = response.choices[0].message.content

        # Save to memories
        supabase.table("memories").insert({
            "user_id": payload.user_id,
            "character_id": payload.character_id,
            "message": payload.message,
            "reply": reply,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        return {
            "character": character["name"],
            "user_message": payload.message,
            "reply": reply
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Save memory manually
@app.post("/memories")
def save_memory(payload: MemoryIn):
    try:
        result = supabase.table("memories").insert({
            "user_id": payload.user_id,
            "character_id": payload.character_id,
            "message": payload.message,
            "reply": payload.reply,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        if result.data:
            return {"status": "success", "memory": result.data[0]}
        else:
            raise HTTPException(status_code=400, detail="Insert failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get chat history for a user + character
@app.get("/memories/{user_id}/{character_id}")
def get_memories(user_id: str, character_id: str):
    try:
        result = supabase.table("memories") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("character_id", character_id) \
            .order("created_at", desc=False) \
            .execute()

        return {"memories": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
