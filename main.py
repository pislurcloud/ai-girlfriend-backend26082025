import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
from typing import Dict, Any, List

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Supabase HTTP client
class SupabaseHTTP:
    def __init__(self, url: str, key: str):
        self.base_url = f"{url}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
    
    async def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{table}",
                json=data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def select(self, table: str, columns: str = "*", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/{table}?select={columns}"
        
        if filters:
            for key, value in filters.items():
                url += f"&{key}=eq.{value}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
    
    async def select_single(self, table: str, columns: str = "*", filters: Dict[str, Any] = None) -> Dict[str, Any]:
        results = await self.select(table, columns, filters)
        if not results:
            return None
        return results[0]

# Initialize Supabase HTTP client
supabase = SupabaseHTTP(SUPABASE_URL, SUPABASE_KEY)

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
        response = await supabase.insert("characters", {
            "name": character.name,
            "persona": character.persona
        })
        return {"status": "success", "data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters")
async def list_characters():
    try:
        response = await supabase.select("characters")
        return {"status": "success", "data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_character(chat: ChatRequest):
    try:
        # Fetch character
        character = await supabase.select_single("characters", "*", {"id": chat.character_id})
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Fetch memories
        memories = await supabase.select("memories", "memory_text", {"character_id": chat.character_id})
        memory_texts = [m["memory_text"] for m in memories] if memories else []

        # System prompt
        system_prompt = f"""
        You are roleplaying as {character['name']}.
        Persona: {character['persona']}
        Relevant memories: {"; ".join(memory_texts) if memory_texts else "No memories yet."}
        """

        # Get response from OpenAI
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat.message}
            ],
        )

        reply = completion.choices[0].message.content

        # Save conversation to memories
        await supabase.insert("memories", {
            "character_id": chat.character_id,
            "memory_text": f"User: {chat.message} | {character['name']}: {reply}",
            "created_at": datetime.utcnow().isoformat()
        })

        return {"status": "success", "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories")
async def create_memory(memory: MemoryCreate):
    try:
        response = await supabase.insert("memories", {
            "character_id": memory.character_id,
            "memory_text": memory.memory_text,
            "created_at": datetime.utcnow().isoformat()
        })
        return {"status": "success", "data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{character_id}")
async def list_memories(character_id: str):
    try:
        response = await supabase.select("memories", "*", {"character_id": character_id})
        return {"status": "success", "data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "AI Characters API is running 🚀"}
