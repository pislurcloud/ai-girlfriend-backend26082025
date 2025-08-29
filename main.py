from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from openai import OpenAI
import os
import jwt
import bcrypt
from typing import Optional, List, Dict, Any

# Load env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")

if not all([OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("Missing one or more environment variables")

# Init clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
security = HTTPBearer(auto_error=False)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Update with your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Models ---------
class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    message: str

class CharacterRequest(BaseModel):
    name: str
    persona: Dict[str, Any]

class FetchMemoriesRequest(BaseModel):
    user_id: str
    character_id: str

class UserRequest(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    token: Optional[str] = None

class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

# --------- Endpoints ---------

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Companion backend running"}

@app.post("/users/register", response_model=UserResponse)
def register_user(req: UserRegisterRequest):
    try:
        # Check if username already exists
        existing = supabase.table("users").select("*").eq("username", req.username).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        existing_email = supabase.table("users").select("*").eq("email", req.email).execute()
        if existing_email.data:
            raise HTTPException(status_code=400, detail="Email already exists")

        # Hash password and create user
        hashed_password = hash_password(req.password)
        result = supabase.table("users").insert({
            "username": req.username,
            "email": req.email,
            "password_hash": hashed_password
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create user")

        user_data = result.data[0]
        token = create_jwt_token(str(user_data["id"]))
        
        return UserResponse(
            id=str(user_data["id"]), 
            username=user_data["username"],
            email=user_data["email"],
            token=token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.post("/users/login", response_model=UserResponse)
def login_user(req: UserLoginRequest):
    try:
        # Find user by username
        result = supabase.table("users").select("*").eq("username", req.username).execute()
        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        user_data = result.data[0]
        
        # Verify password
        if not verify_password(req.password, user_data["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Create JWT token
        token = create_jwt_token(str(user_data["id"]))
        
        return UserResponse(
            id=str(user_data["id"]), 
            username=user_data["username"],
            email=user_data.get("email"),
            token=token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.post("/users", response_model=UserResponse)
def create_user(req: UserRequest):
    # Legacy endpoint for backward compatibility
    try:
        # Check if username already exists
        existing = supabase.table("users").select("*").eq("username", req.username).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Username already exists")

        # Insert new user (no password for legacy support)
        result = supabase.table("users").insert({"username": req.username}).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create user")

        user_data = result.data[0]
        return UserResponse(id=str(user_data["id"]), username=user_data["username"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users")
def get_user(username: str):
    """Legacy endpoint - get user by username"""
    try:
        result = supabase.table("users").select("*").eq("username", username).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = result.data[0]
        return {"id": str(user_data["id"]), "username": user_data["username"]}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users/me")
def read_users_me():
    return {"message": "Auth disabled for now"}

# --------- Character Management with User Association ---------
def create_character(req: CharacterRequest):
    try:
        result = supabase.table("characters").insert({
            "name": req.name,
            "persona": req.persona,
            "user_id": req.user_id  # Associate character with user
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create character")
            
        return {"character": result.data[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/characters")
def get_all_characters():
    """Get all characters (for backward compatibility)"""
    try:
        result = supabase.table("characters").select("*").execute()
        return {"characters": result.data or []}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/characters/user/{user_id}")
def get_user_characters(user_id: str):
    """Get characters for a specific user"""
    try:
        result = supabase.table("characters").select("*").eq("user_id", user_id).execute()
        return {"characters": result.data or []}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/memories")
def fetch_memories(req: FetchMemoriesRequest):
    """Fetch conversation history for a user-character pair"""
    try:
        result = supabase.table("memories").select("*")\
            .eq("user_id", req.user_id)\
            .eq("character_id", req.character_id)\
            .order("created_at", desc=False)\
            .execute()
        
        return result.data or []
    
    except Exception as e:
        print(f"ERROR fetching memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Retrieve character persona
        character_result = supabase.table("characters").select("*").eq("id", req.character_id).execute()
        if not character_result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        
        character = character_result.data[0]
        persona = character.get("persona", {})
        
        # Build system prompt
        character_name = persona.get('name', character.get('name', 'AI Companion'))
        character_style = persona.get('style', 'friendly and supportive')
        character_bio = persona.get('bio', '')
        
        system_prompt = f"""You are {character_name}, an AI companion. 
        Your personality style: {character_style}
        {f"Background: {character_bio}" if character_bio else ""}
        
        Guidelines:
        - Be conversational and engaging
        - Stay in character with your personality style
        - Keep responses reasonably concise but meaningful
        - Be supportive and understanding"""

        # Get recent conversation context (last 10 messages)
        try:
            recent_memories = supabase.table("memories").select("message, response")\
                .eq("user_id", req.user_id)\
                .eq("character_id", req.character_id)\
                .order("created_at", desc=True)\
                .limit(5)\
                .execute()
            
            conversation_history = []
            if recent_memories.data:
                # Reverse to get chronological order
                for memory in reversed(recent_memories.data):
                    if memory.get("message"):
                        conversation_history.append({"role": "user", "content": memory["message"]})
                    if memory.get("response"):
                        conversation_history.append({"role": "assistant", "content": memory["response"]})
        
        except Exception as e:
            print(f"Warning: Could not load conversation history: {str(e)}")
            conversation_history = []

        # Prepare messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": req.message})

        # Generate AI response
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.8
            )
            reply = completion.choices[0].message.content
        
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            reply = "I'm sorry, I'm having trouble thinking right now. Could you try asking me again?"

        # Save conversation to memories
        try:
            memory_data = {
                "user_id": req.user_id,
                "character_id": req.character_id,
                "message": req.message,
                "response": reply,
                "created_at": datetime.utcnow().isoformat()
            }
            supabase.table("memories").insert(memory_data).execute()
        except Exception as e:
            print(f"Warning: Could not save conversation: {str(e)}")

        return {"reply": reply}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")

# Optional: Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Test database connection
        supabase.table("users").select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

# Optional: Get character by ID
@app.get("/characters/{character_id}")
def get_character(character_id: str):
    try:
        result = supabase.table("characters").select("*").eq("id", character_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        return {"character": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
