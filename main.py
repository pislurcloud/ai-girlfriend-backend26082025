from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
import os

# --- Initialize FastAPI ---
app = FastAPI()

# --- Environment Variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not OPENAI_API_KEY:
    raise Exception("Missing environment variables. Please set SUPABASE_URL, SUPABASE_SERVICE_KEY, and OPENAI_API_KEY.")

# --- Clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Request Models ---
class CharacterCreate(BaseModel):
    name: str
    persona: dict  # expects JSONB in Supabase

class ChatRequest(BaseModel):
    character_id: str
    message: str

# --- Characters Endpoints ---
@app.get("/characters")
async def list_characters():
    """Fetch all characters from Supabase"""
    try:
        response = supabase.table("characters").select("*").execute()
        return {"characters": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters")
async def create_character(character: CharacterCreate):
    """Create a new character"""
    try:
        response = supabase.table("characters").insert({
            "name": character.name,
            "persona": character.persona
        }).execute()
        return {"character": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(request: ChatRequest):
    """Send a message to a character and get an AI response"""
    try:
        # Fetch character from Supabase
        response = supabase.table("characters").select("*").eq("id", request.character_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Character not found")

        character = response.data[0]
        name = character["name"]
        persona = character.get("persona", {})

        # Build system prompt
        persona_text = f"You are {name}, an AI companion."
        if "description" in persona:
            persona_text += f" {persona['description']}"
        if "style" in persona:
            persona_text += f" Your communication style is {persona['style']}."
        if "likes" in persona:
            persona_text += f" You enjoy {', '.join(persona['likes'])}."

        # Call OpenAI
        chat_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": persona_text},
                {"role": "user", "content": request.message}
            ],
            max_tokens=200
        )

        reply = chat_response.choices[0].message.content
        return {"reply": reply}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
