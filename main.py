import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
from typing import List, Dict, Any

# Initialize FastAPI
app = FastAPI()

# Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# OpenAI client (no proxy arg!)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Request Models
# ----------------------------
class CharacterCreate(BaseModel):
    name: str
    persona: Dict[str, Any]

class ChatRequest(BaseModel):
    character_id: str
    user_id: str
    message: str

# ----------------------------
# Characters Endpoints
# ----------------------------
@app.post("/characters")
def create_character(character: CharacterCreate):
    try:
        data = {
            "name": character.name,
            "persona": character.persona
        }
        response = supabase.table("characters").insert(data).execute()
        return {"status": "success", "character": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/characters")
def list_characters():
    try:
        response = supabase.table("characters").select("*").execute()
        return {"characters": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Memories Endpoints
# ----------------------------
@app.get("/memories/{character_id}")
def get_memories(character_id: str):
    try:
        response = supabase.table("memories").select("*").eq("character_id", character_id).execute()
        return {"memories": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Chat Endpoint
# ----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Fetch character
        character_resp = supabase.table("characters").select("*").eq("id", request.character_id).execute()
        if not character_resp.data:
            raise HTTPException(status_code=404, detail="Character not found")
        character = character_resp.data[0]

        # Fetch past memories
        memories_resp = supabase.table("memories").select("*").eq("character_id", request.character_id).order("created_at").execute()
        past_memories = memories_resp.data

        # Construct system prompt from persona + memories
        persona = character.get("persona", {})
        persona_text = "\n".join([f"{k}: {v}" for k, v in persona.items()])
        memory_text = "\n".join([f"{m['user_id']}: {m['message']}" for m in past_memories])

        system_prompt = f"""
        You are {character['name']}, an AI girlfriend with the following traits:
        {persona_text}

        Past conversation history:
        {memory_text}
        """

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ]
        )

        reply = response.choices[0].message.content

        # Save memory
        supabase.table("memories").insert({
            "character_id": request.character_id,
            "user_id": request.user_id,
            "message": request.message,
            "response": reply
        }).execute()

        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
