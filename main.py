from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from openai import OpenAI
import os
import jwt
import bcrypt
import uuid
import requests
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

SECRET_KEY = os.getenv("JWT_SECRET", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def create_jwt_token(user_id: str):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": user_id, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Image generation and storage functions
def generate_character_prompt(character_data: Dict[str, Any]) -> str:
    """Generate a detailed DALL-E prompt from character data"""
    name = character_data.get('name', 'person')
    appearance = character_data.get('appearance', {})
    persona = character_data.get('persona', {})
    
    # Build prompt components
    age_desc = f"{appearance.get('age', '25')} years old" if appearance.get('age') else "young adult"
    gender = appearance.get('gender', 'person')
    hair = appearance.get('hair_color', 'brown') + " hair"
    style = appearance.get('style', 'casual modern')
    clothing = appearance.get('clothing', 'stylish outfit')
    mood = persona.get('style', 'friendly').split()[0] if persona.get('style') else 'friendly'
    
    prompt = f"Professional portrait of a {age_desc} {gender} with {hair}, {mood} expression, wearing {clothing}, {style} aesthetic, high quality, photorealistic, soft lighting, clean background"
    
    return prompt

async def generate_and_store_character_image(character_data: Dict[str, Any], user_id: str) -> Optional[str]:
    """Generate character image using DALL-E and store in Supabase"""
    try:
        # Generate image with DALL-E 3
        prompt = generate_character_prompt(character_data)
        print(f"Generating image with prompt: {prompt}")
        
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        
        # Download the image
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            raise Exception("Failed to download generated image")
        
        # Create unique filename
        character_name = character_data.get('name', 'character').replace(' ', '_').lower()
        filename = f"characters/{user_id}/{character_name}_{uuid.uuid4().hex[:8]}.png"
        
        # Upload to Supabase Storage
        result = supabase.storage.from_("character-images").upload(
            filename, 
            img_response.content,
            {"content-type": "image/png"}
        )
        
        if result.data:
            # Get public URL
            public_url = supabase.storage.from_("character-images").get_public_url(filename)
            return public_url.data.get('publicUrl') if public_url.data else None
        else:
            print(f"Upload failed: {result}")
            return None
            
    except Exception as e:
        print(f"Error generating/storing character image: {str(e)}")
        return None

async def generate_chat_image(prompt: str, user_id: str, character_id: str) -> Optional[str]:
    """Generate image for chat based on user request"""
    try:
        # Generate image with DALL-E 3
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=f"{prompt}, high quality, detailed, beautiful",
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        
        # Download the image
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            raise Exception("Failed to download generated image")
        
        # Create unique filename for chat image
        filename = f"chat-images/{user_id}/{character_id}/{uuid.uuid4().hex}.png"
        
        # Upload to Supabase Storage
        result = supabase.storage.from_("character-images").upload(
            filename, 
            img_response.content,
            {"content-type": "image/png"}
        )
        
        if result.data:
            public_url = supabase.storage.from_("character-images").get_public_url(filename)
            return public_url.data.get('publicUrl') if public_url.data else None
        else:
            return None
            
    except Exception as e:
        print(f"Error generating chat image: {str(e)}")
        return None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Enhanced Models ---------
class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    message: str

class EnhancedCharacterRequest(BaseModel):
    user_id: str
    name: str
    persona: Dict[str, Any]
    appearance: Dict[str, Any] = {}
    generate_avatar: bool = True

class CharacterRequest(BaseModel):
    user_id: str
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

class GenerateImageRequest(BaseModel):
    user_id: str
    character_id: str
    prompt: str

# FIXED: Added proper request model for avatar generation
class GenerateAvatarRequest(BaseModel):
    user_id: str

# --------- Endpoints ---------

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Companion backend running with image generation"}

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
        existing = supabase.table("users").select("*").eq("username", req.username).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Username already exists")

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

# --------- Enhanced Character Management ---------
@app.post("/characters/enhanced")
async def create_enhanced_character(req: EnhancedCharacterRequest):
    """Create character with detailed appearance and optional avatar generation"""
    try:
        # Prepare character data
        character_data = {
            "name": req.name,
            "persona": req.persona,
            "appearance": req.appearance,
            "user_id": req.user_id,
            "avatar_url": None
        }
        
        # Generate avatar if requested
        if req.generate_avatar:
            print(f"Generating avatar for character: {req.name}")
            avatar_url = await generate_and_store_character_image(
                {
                    "name": req.name,
                    "persona": req.persona,
                    "appearance": req.appearance
                },
                req.user_id
            )
            character_data["avatar_url"] = avatar_url
            print(f"Generated avatar URL: {avatar_url}")
        
        # Insert character into database
        result = supabase.table("characters").insert(character_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create character")
            
        return {"character": result.data[0]}
    
    except Exception as e:
        print(f"Error creating enhanced character: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Character creation error: {str(e)}")

@app.post("/characters")
def create_character(req: CharacterRequest):
    """Legacy character creation endpoint"""
    try:
        result = supabase.table("characters").insert({
            "name": req.name,
            "persona": req.persona,
            "user_id": req.user_id,
            "appearance": {},
            "avatar_url": None
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create character")
            
        return {"character": result.data[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# FIXED: Proper request body approach for avatar generation
@app.post("/characters/{character_id}/generate-avatar")
async def generate_character_avatar(character_id: str, req: GenerateAvatarRequest):
    """Generate avatar for existing character using request body"""
    try:
        print(f"Generating avatar for character {character_id}, user {req.user_id}")
        
        # Get character data
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", req.user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        character = result.data[0]
        print(f"Found character: {character.get('name')}")
        
        # Generate avatar
        avatar_url = await generate_and_store_character_image(character, req.user_id)
        
        if avatar_url:
            print(f"Generated avatar URL: {avatar_url}")
            # Update character with avatar URL
            update_result = supabase.table("characters").update({
                "avatar_url": avatar_url
            }).eq("id", character_id).execute()
            
            if update_result.data:
                return {"avatar_url": avatar_url, "character": update_result.data[0]}
            else:
                return {"avatar_url": avatar_url, "character": character}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate avatar")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating avatar: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Avatar generation error: {str(e)}")

@app.post("/generate-chat-image")
async def generate_chat_image_endpoint(req: GenerateImageRequest):
    """Generate image during chat conversation"""
    try:
        image_url = await generate_chat_image(req.prompt, req.user_id, req.character_id)
        
        if image_url:
            return {"image_url": image_url, "prompt": req.prompt}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate image")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating chat image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

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
async def chat(req: ChatRequest):
    try:
        print(f"Chat request: user_id={req.user_id}, character_id={req.character_id}, message={req.message[:50]}...")
        
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
        - Be supportive and understanding
        - If the user asks you to generate, create, show, or make an image, respond with "I'll create that image for you!" and include [GENERATE_IMAGE: detailed description] at the end of your message"""

        # Get recent conversation context (last 10 messages)
        try:
            recent_memories = supabase.table("memories").select("message, response, image_url")\
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
            print(f"AI replied: {reply[:100]}...")
        
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            reply = "I'm sorry, I'm having trouble thinking right now. Could you try asking me again?"

        # FIXED: Better image generation detection and handling
        generated_image_url = None
        if "[GENERATE_IMAGE:" in reply:
            try:
                print("Detected image generation request in AI response")
                # Extract image prompt
                start = reply.find("[GENERATE_IMAGE:") + len("[GENERATE_IMAGE:")
                end = reply.find("]", start)
                if end > start:
                    image_prompt = reply[start:end].strip()
                    print(f"Extracted image prompt: {image_prompt}")
                    
                    # Remove the generate image instruction from the reply
                    reply = reply.replace(f"[GENERATE_IMAGE:{image_prompt}]", "").strip()
                    
                    # Generate the image
                    print("Starting image generation...")
                    generated_image_url = await generate_chat_image(image_prompt, req.user_id, req.character_id)
                    print(f"Generated chat image URL: {generated_image_url}")
                    
            except Exception as e:
                print(f"Error generating chat image: {str(e)}")
                # Don't fail the entire chat if image generation fails

        # Save conversation to memories
        try:
            memory_data = {
                "user_id": req.user_id,
                "character_id": req.character_id,
                "message": req.message,
                "response": reply,
                "image_url": generated_image_url,
                "created_at": datetime.utcnow().isoformat()
            }
            memory_result = supabase.table("memories").insert(memory_data).execute()
            print(f"Saved conversation to memories: {memory_result.data is not None}")
        except Exception as e:
            print(f"Warning: Could not save conversation: {str(e)}")

        response_data = {"reply": reply}
        if generated_image_url:
            response_data["image_url"] = generated_image_url
            
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")

# --------- Storage Management ---------
@app.post("/setup-storage")
async def setup_storage():
    """Initialize Supabase storage bucket for character images"""
    try:
        # Create bucket if it doesn't exist
        bucket_result = supabase.storage.create_bucket("character-images", {"public": True})
        return {"message": "Storage bucket created successfully", "result": bucket_result}
    except Exception as e:
        # Bucket might already exist
        return {"message": "Storage setup complete (bucket may already exist)", "error": str(e)}

# --------- Health Check ---------
@app.get("/health")
def health_check():
    try:
        # Test database connection
        supabase.table("users").select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected", "features": ["chat", "images"], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

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
