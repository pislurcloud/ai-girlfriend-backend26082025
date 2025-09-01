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
import base64
from io import BytesIO
from pydub import AudioSegment
import speech_recognition as sr

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
    """Generate a detailed DALL-E prompt from character data with enhanced realism"""
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
    
    # Enhanced prompt for maximum realism
    prompt = f"Ultra-realistic professional headshot photograph of a {age_desc} {gender} with {hair}, {mood} expression, wearing {clothing}, {style} aesthetic, shot with a Canon EOS R5, 85mm lens, natural studio lighting, shallow depth of field, photojournalism style, real human features, skin texture detail, natural lighting, professional photography, 4K quality"
    
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
        try:
            result = supabase.storage.from_("character-images").upload(
                filename, 
                img_response.content,
                {"content-type": "image/png"}
            )
            print(f"Upload result type: {type(result)}")
            print(f"Upload result: {result}")
            
            # Check if upload was successful (different response structure)
            if hasattr(result, 'path') or (hasattr(result, 'data') and result.data):
                # Get public URL
                public_url_response = supabase.storage.from_("character-images").get_public_url(filename)
                print(f"Public URL response: {public_url_response}")
                
                # Handle different response structures
                if hasattr(public_url_response, 'data'):
                    return public_url_response.data.get('publicUrl')
                elif hasattr(public_url_response, 'publicUrl'):
                    return public_url_response.publicUrl
                elif isinstance(public_url_response, str):
                    return public_url_response
                else:
                    print(f"Unexpected public URL response format: {type(public_url_response)}")
                    return None
            else:
                print(f"Upload failed: {result}")
                return None
        except Exception as upload_error:
            print(f"Upload exception: {str(upload_error)}")
            return None
            
    except Exception as e:
        print(f"Error generating/storing character image: {str(e)}")
        return None

async def generate_chat_image(prompt: str, user_id: str, character_id: str) -> Optional[str]:
    """Generate image for chat based on user request with enhanced realism"""
    try:
        # Enhance prompt for better realism based on content type
        enhanced_prompt = enhance_image_prompt(prompt)
        print(f"Enhanced chat image prompt: {enhanced_prompt}")
        
        # Generate image with DALL-E 3
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="hd",  # Use HD quality for better results
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
        try:
            result = supabase.storage.from_("character-images").upload(
                filename, 
                img_response.content,
                {"content-type": "image/png"}
            )
            print(f"Chat image upload result: {result}")
            
            # Check if upload was successful
            if hasattr(result, 'path') or (hasattr(result, 'data') and result.data):
                # Get public URL
                public_url_response = supabase.storage.from_("character-images").get_public_url(filename)
                
                # Handle different response structures
                if hasattr(public_url_response, 'data'):
                    return public_url_response.data.get('publicUrl')
                elif hasattr(public_url_response, 'publicUrl'):
                    return public_url_response.publicUrl
                elif isinstance(public_url_response, str):
                    return public_url_response
                else:
                    return None
            else:
                return None
        except Exception as upload_error:
            print(f"Chat image upload exception: {str(upload_error)}")
            return None
            
    except Exception as e:
        print(f"Error generating chat image: {str(e)}")
        return None

def enhance_image_prompt(user_prompt: str) -> str:
    """Enhance user prompts for better DALL-E 3 realism"""
    prompt_lower = user_prompt.lower()
    
    # Detect content type and add appropriate enhancement
    if any(word in prompt_lower for word in ['person', 'people', 'man', 'woman', 'human', 'face', 'portrait']):
        return f"Ultra-realistic professional photograph of {user_prompt}, shot with Canon EOS R5, 85mm lens, natural studio lighting, photojournalism style, high detail, 4K quality"
    
    elif any(word in prompt_lower for word in ['food', 'meal', 'dish', 'cooking', 'restaurant', 'kitchen']):
        return f"Professional food photography of {user_prompt}, appetizing, restaurant quality, natural lighting, macro lens, food styling, high resolution"
    
    elif any(word in prompt_lower for word in ['place', 'location', 'city', 'landscape', 'building', 'room', 'house']):
        return f"High-resolution travel photography of {user_prompt}, professional landscape photography, natural lighting, wide angle lens, realistic, detailed"
    
    elif any(word in prompt_lower for word in ['animal', 'dog', 'cat', 'bird', 'wildlife']):
        return f"Professional wildlife photography of {user_prompt}, National Geographic style, natural habitat, high detail, realistic"
    
    else:
        # General objects or abstract concepts
        return f"High-quality realistic photograph of {user_prompt}, professional photography, natural lighting, detailed, 4K resolution"

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
    audio: Optional[str] = None  # Base64 encoded audio

class EnhancedCharacterRequest(BaseModel):
    user_id: str
    name: str
    persona: Dict[str, Any]
    appearance: Dict[str, Any] = {}
    voice_settings: Dict[str, Any] = {
        "voiceId": "en-US-Studio-O",
        "speed": 1.0,
        "pitch": 1.0
    }
    generate_avatar: bool = True

class CharacterRequest(BaseModel):
    user_id: str
    name: str
    persona: Dict[str, Any]
    voice_settings: Optional[Dict[str, Any]] = None

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
            "voice_settings": req.voice_settings,
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
            "voice_settings": req.voice_settings,
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
        # If audio is provided, transcribe it
        if req.audio:
            try:
                # Convert base64 to audio file
                audio_data = base64.b64decode(req.audio.split(',')[1])  # Remove data:audio/wav;base64, prefix
                audio_file = BytesIO(audio_data)
                
                # Convert to WAV format
                audio_segment = AudioSegment.from_file(audio_file, format="wav")
                wav_data = BytesIO()
                audio_segment.export(wav_data, format="wav")
                wav_data.seek(0)
                
                # Transcribe using speech recognition
                r = sr.Recognizer()
                with sr.AudioFile(wav_data) as source:
                    audio_data = r.record(source)
                    transcribed_text = r.recognize_google(audio_data)
                    req.message = transcribed_text
            except Exception as e:
                print(f"Error transcribing audio: {str(e)}")
                raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Rest of your existing chat logic...
        character = supabase.table("characters").select("*").eq("id", req.character_id).single().execute()
        character = character.data
        
        # Get conversation history
        messages = supabase.table("memories") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .eq("character_id", req.character_id) \
            .order("created_at", desc=True) \
            .limit(5) \
            .execute()
            
        # Prepare conversation history for the AI
        conversation_history = [
            {"role": "system", "content": f"You are {character['name']}. {character['persona']}"}
        ]
        
        # Add previous messages to context
        for msg in reversed(messages.data):
            conversation_history.append({"role": "user", "content": msg["message"]})
            if msg["response"]:
                conversation_history.append({"role": "assistant", "content": msg["response"]})
                
        # Add current message
        conversation_history.append({"role": "user", "content": req.message})
        
        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=500,
            temperature=0.7
        )
        
        reply = response.choices[0].message.content
        
        # Save to database
        memory_data = {
            "user_id": req.user_id,
            "character_id": req.character_id,
            "message": req.message,
            "response": reply,
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("memories").insert(memory_data).execute()
        
        return {"reply": reply}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Add these endpoints to your main.py file (These changes added recently, remove them if we get error)

# --------- Character Management Enhancements ---------

class UpdateCharacterRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    persona: Optional[Dict[str, Any]] = None
    appearance: Optional[Dict[str, Any]] = None
    voice_settings: Optional[Dict[str, Any]] = None
    regenerate_avatar: bool = False

class PasswordResetRequest(BaseModel):
    username: str
    email: str

class PasswordUpdateRequest(BaseModel):
    username: str
    new_password: str
    reset_token: str

@app.put("/characters/{character_id}")
async def update_character(character_id: str, req: UpdateCharacterRequest):
    """Update character details and optionally regenerate avatar"""
    try:
        # Verify character ownership
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", req.user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        current_character = result.data[0]
        
        # Prepare update data
        update_data = {}
        if req.name is not None:
            update_data["name"] = req.name
        if req.persona is not None:
            update_data["persona"] = req.persona
        if req.appearance is not None:
            update_data["appearance"] = req.appearance
        if req.voice_settings is not None:
            update_data["voice_settings"] = req.voice_settings
        
        # Regenerate avatar if requested or if appearance changed
        if req.regenerate_avatar or req.appearance is not None:
            print(f"Regenerating avatar for character {character_id}")
            
            # Merge current data with updates for avatar generation
            character_data = {
                "name": req.name or current_character.get("name"),
                "persona": req.persona or current_character.get("persona", {}),
                "appearance": req.appearance or current_character.get("appearance", {})
            }
            
            avatar_url = await generate_and_store_character_image(character_data, req.user_id)
            if avatar_url:
                update_data["avatar_url"] = avatar_url
                
                # Delete old avatar if it exists
                if current_character.get("avatar_url"):
                    try:
                        # Extract filename from URL for deletion
                        old_url = current_character["avatar_url"]
                        if "character-images" in old_url:
                            filename = old_url.split("character-images/")[1]
                            supabase.storage.from_("character-images").remove([filename])
                    except Exception as e:
                        print(f"Warning: Could not delete old avatar: {str(e)}")
        
        # Update character in database
        if update_data:
            update_result = supabase.table("characters").update(update_data).eq("id", character_id).execute()
            if not update_result.data:
                raise HTTPException(status_code=500, detail="Failed to update character")
            return {"character": update_result.data[0]}
        else:
            return {"character": current_character}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating character: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Character update error: {str(e)}")

@app.delete("/characters/{character_id}")
async def delete_character(character_id: str, user_id: str = Query(...)):
    """Delete character and associated data"""
    try:
        # Verify character ownership
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        character = result.data[0]
        
        # Delete character avatar from storage
        if character.get("avatar_url"):
            try:
                old_url = character["avatar_url"]
                if "character-images" in old_url:
                    filename = old_url.split("character-images/")[1]
                    supabase.storage.from_("character-images").remove([filename])
            except Exception as e:
                print(f"Warning: Could not delete avatar: {str(e)}")
        
        # Delete all chat images for this character
        try:
            # Get all memories with images for this character
            memories_with_images = supabase.table("memories").select("image_url")\
                .eq("character_id", character_id)\
                .eq("user_id", user_id)\
                .not_("image_url", "is", None)\
                .execute()
            
            if memories_with_images.data:
                files_to_delete = []
                for memory in memories_with_images.data:
                    if memory.get("image_url") and "character-images" in memory["image_url"]:
                        filename = memory["image_url"].split("character-images/")[1]
                        files_to_delete.append(filename)
                
                if files_to_delete:
                    supabase.storage.from_("character-images").remove(files_to_delete)
        except Exception as e:
            print(f"Warning: Could not delete chat images: {str(e)}")
        
        # Delete conversation memories
        supabase.table("memories").delete().eq("character_id", character_id).eq("user_id", user_id).execute()
        
        # Delete character
        delete_result = supabase.table("characters").delete().eq("id", character_id).execute()
        
        return {"message": "Character deleted successfully", "deleted_character": character}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting character: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Character deletion error: {str(e)}")

# --------- Password Reset Functionality ---------

def generate_reset_token(user_id: str) -> str:
    """Generate a password reset token"""
    expire = datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    to_encode = {"sub": user_id, "type": "reset", "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_reset_token(token: str) -> Optional[str]:
    """Verify reset token and return user_id"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "reset":
            return None
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

@app.post("/users/request-password-reset")
async def request_password_reset(req: PasswordResetRequest):
    """Request password reset - generates reset token"""
    try:
        # Find user by username and email
        result = supabase.table("users").select("*")\
            .eq("username", req.username)\
            .eq("email", req.email)\
            .execute()
        
        if not result.data:
            # For security, don't reveal if user exists
            return {"message": "If an account with that username and email exists, a reset link has been sent."}
        
        user_data = result.data[0]
        reset_token = generate_reset_token(str(user_data["id"]))
        
        # In a real app, you'd send this via email
        # For now, we'll return it (remove this in production!)
        return {
            "message": "Password reset token generated",
            "reset_token": reset_token,  # Remove this in production
            "note": "In production, this would be sent via email"
        }
    
    except Exception as e:
        print(f"Password reset request error: {str(e)}")
        raise HTTPException(status_code=500, detail="Password reset request failed")

@app.post("/users/reset-password")
async def reset_password(req: PasswordUpdateRequest):
    """Reset password using reset token"""
    try:
        # Verify reset token
        user_id = verify_reset_token(req.reset_token)
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
        # Find user
        result = supabase.table("users").select("*").eq("id", user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = result.data[0]
        
        # Verify username matches
        if user_data["username"] != req.username:
            raise HTTPException(status_code=400, detail="Username mismatch")
        
        # Update password
        new_hashed_password = hash_password(req.new_password)
        update_result = supabase.table("users").update({
            "password_hash": new_hashed_password
        }).eq("id", user_id).execute()
        
        if not update_result.data:
            raise HTTPException(status_code=500, detail="Failed to update password")
        
        return {"message": "Password reset successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Password reset error: {str(e)}")
        raise HTTPException(status_code=500, detail="Password reset failed")

# --------- User Profile Management ---------

class UpdateProfileRequest(BaseModel):
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None

@app.put("/users/{user_id}/profile")
async def update_user_profile(user_id: str, req: UpdateProfileRequest):
    """Update user profile information"""
    try:
        # Verify user ownership (in real app, verify JWT token matches user_id)
        if req.user_id != user_id:
            raise HTTPException(status_code=403, detail="Cannot update another user's profile")
        
        update_data = {}
        if req.username is not None:
            # Check if new username is available
            existing = supabase.table("users").select("*").eq("username", req.username).neq("id", user_id).execute()
            if existing.data:
                raise HTTPException(status_code=400, detail="Username already taken")
            update_data["username"] = req.username
        
        if req.email is not None:
            # Check if new email is available
            existing = supabase.table("users").select("*").eq("email", req.email).neq("id", user_id).execute()
            if existing.data:
                raise HTTPException(status_code=400, detail="Email already in use")
            update_data["email"] = req.email
        
        if update_data:
            result = supabase.table("users").update(update_data).eq("id", user_id).execute()
            if not result.data:
                raise HTTPException(status_code=500, detail="Failed to update profile")
            
            return {"user": result.data[0], "message": "Profile updated successfully"}
        else:
            return {"message": "No changes to update"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Profile update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Profile update failed")

# --------- Enhanced Analytics & Insights ---------

@app.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user statistics and insights"""
    try:
        # Get character count
        chars_result = supabase.table("characters").select("id").eq("user_id", user_id).execute()
        character_count = len(chars_result.data) if chars_result.data else 0
        
        # Get total message count
        messages_result = supabase.table("memories").select("id").eq("user_id", user_id).execute()
        total_messages = len(messages_result.data) if messages_result.data else 0
        
        # Get images generated count
        images_result = supabase.table("memories").select("id").eq("user_id", user_id).not_("image_url", "is", None).execute()
        images_generated = len(images_result.data) if images_result.data else 0
        
        # Get most active character
        if character_count > 0:
            # Count messages per character
            all_memories = supabase.table("memories").select("character_id").eq("user_id", user_id).execute()
            if all_memories.data:
                char_message_counts = {}
                for memory in all_memories.data:
                    char_id = memory["character_id"]
                    char_message_counts[char_id] = char_message_counts.get(char_id, 0) + 1
                
                if char_message_counts:
                    most_active_char_id = max(char_message_counts, key=char_message_counts.get)
                    char_result = supabase.table("characters").select("name").eq("id", most_active_char_id).execute()
                    most_active_character = char_result.data[0]["name"] if char_result.data else "Unknown"
                else:
                    most_active_character = None
            else:
                most_active_character = None
        else:
            most_active_character = None
        
        return {
            "character_count": character_count,
            "total_messages": total_messages,
            "images_generated": images_generated,
            "most_active_character": most_active_character
        }
    
    except Exception as e:
        print(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user stats")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
