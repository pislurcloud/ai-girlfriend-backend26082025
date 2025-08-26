from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
#import openai
from openai import OpenAI
import os

# Load env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("Missing one or more environment variables")

# Init clients
#openai.api_key = OPENAI_API_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

# --------- Models ---------
class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    message: str

class CharacterRequest(BaseModel):
    name: str
    persona: dict

class MemoryRequest(BaseModel):
    user_id: str
    character_id: str
    content: str

# --------- Endpoints ---------

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Girlfriend backend running"}

@app.post("/characters")
def create_character(req: CharacterRequest):
    data, error = supabase.table("characters").insert({
        "name": req.name,
        "persona": req.persona
    }).execute()
    if error:
        raise HTTPException(status_code=500, detail=str(error))
    return {"character": data.data[0]}

@app.post("/memories")
def create_memory(req: MemoryRequest):
    # Generate embedding
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=req.content
    )
    vector = emb.data[0].embedding

    data, error = supabase.table("memories").insert({
        "user_id": req.user_id,
        "character_id": req.character_id,
        "content": req.content,
        "embedding": vector
    }).execute()

    if error:
        raise HTTPException(status_code=500, detail=str(error))
    return {"memory": data.data[0]}



client = OpenAI(api_key=OPENAI_API_KEY)  # or use environment variable

@app.post("/chat")
def chat(req: ChatRequest):
    # Retrieve character persona
    character = supabase.table("characters").select("*").eq("id", req.character_id).execute()
    if not character.data:
        raise HTTPException(status_code=404, detail="Character not found")
    persona = character.data[0]["persona"]

    # Simple context prompt
    system_prompt = f"You are {persona.get('name', 'an AI girlfriend')}. " \
                    f"Your style: {persona.get('style', 'kind and supportive')}."

    # Create chat completion using new SDK
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    )

    reply = completion.choices[0].message.content
    return {"reply": reply}
