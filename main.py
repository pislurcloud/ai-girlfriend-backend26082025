import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI
from datetime import datetime

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI()

# Allow CORS for local React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic Models
# ---------------------------

class CharacterCreate(BaseModel):
    name: str
    persona: dict

class ChatRequest(BaseModel):
    character_id: str
    message: str

class MemoryCreate(BaseModel):
    character_id: str
    memory_text: str

# ---------------------------
# Endpoints
# ---------------------------

@app.post("/characters")
async def create_character(character: CharacterCreate):
    try:
        response = (
            supabase.table("characters")
            .insert({
                "name": character.name,
                "persona": character.persona
            })
            .execute()
        )
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters")
async def list_characters():
    try:
        response = supabase.table("characters").select("*").execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_character(chat: ChatRequest):
    try:
        # Fetch character
        character_res = (
            supabase.table("characters")
            .select("*")
            .eq("id", chat.character_id)
            .single()
            .execute()
        )
        if not character_res.data:
            raise HTTPException(status_code=404, detail="Character not found")

        character = character_res.data

        # Fetch memories
        memories_res = (
            supabase.table("memories")
            .select("memory_text")
            .eq("character_id", chat.character_id)
            .execute()
        )
        memories = [m["memory_text"] for m in memories_res.data] if memories_res.data else []

        # System prompt
        system_prompt = f"""
        You are roleplaying as {character['name']}.
        Persona: {character['persona']}
        Relevant memories: {"; ".join(memories) if memories else "No memories yet."}
        """

        # Get response from OpenAI
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat.message}
            ],
        )

        reply = completion.choices[0].message["content"]

        # Save conversation to memories for now (optional future refinement)
        supabase.table("memories").insert({
            "character_id": chat.character_id,
            "memory_text": f"User: {chat.message} | {character['name']}: {reply}",
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        return {"status": "success", "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories")
async def create_memory(memory: MemoryCreate):
    try:
        response = (
            supabase.table("memories")
            .insert({
                "character_id": memory.character_id,
                "memory_text": memory.memory_text,
                "created_at": datetime.utcnow().isoformat()
            })
            .execute()
        )
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{character_id}")
async def list_memories(character_id: str):
    try:
        response = (
            supabase.table("memories")
            .select("*")
            .eq("character_id", character_id)
            .execute()
        )
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "AI Characters API is running 🚀"}
