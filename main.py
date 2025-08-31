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

# Add these to your main.py for Phase 3 Voice Implementation

import base64
import io
from pydub import AudioSegment
from pydub.playback import play
import tempfile

# Load env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")

if not all([OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("Missing one or more environment variables")

# Enhanced logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        - If the user asks you to generate, create, show, or make an image, respond with "I'll create that image for you!" and include [GENERATE_IMAGE: detailed description] at the end of your message
        - For image prompts, be very specific about style:
          * For people: "Ultra-realistic photograph, professional photography, natural lighting"
          * For food: "Professional food photography, appetizing, restaurant quality"
          * For places: "High-resolution photograph, travel photography, realistic"
          * For objects: "Product photography, high-quality, realistic lighting"
          * Avoid words like "painting", "artwork", "illustration", "digital art"
        """

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

# Add these endpoints to your main.py file (These changes added recently, remove them if we get error)

# --------- Character Management Enhancements ---------

class UpdateCharacterRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    persona: Optional[Dict[str, Any]] = None
    appearance: Optional[Dict[str, Any]] = None
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


 --------- Enhanced Voice Models with Validation ---------

class VoiceRequest(BaseModel):
    user_id: str
    character_id: str
    audio_data: str  # Base64 encoded audio
    format: str = "webm"
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if not v or len(v) < 100:  # Basic validation
            raise ValueError('Audio data appears to be empty or too short')
        try:
            # Test if it's valid base64
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 audio data')
        return v
    
    @validator('format')
    def validate_format(cls, v):
        allowed_formats = ['webm', 'ogg', 'wav', 'mp3', 'm4a']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Unsupported audio format. Allowed: {allowed_formats}')
        return v.lower()

class TextToSpeechRequest(BaseModel):
    text: str
    character_id: str
    voice_style: Optional[str] = None
    
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        if len(v) > 4000:  # OpenAI TTS limit
            raise ValueError('Text too long for speech synthesis (max 4000 characters)')
        return v.strip()

class VoiceSettingsRequest(BaseModel):
    character_id: str
    user_id: str
    voice_config: Dict[str, Any]
    
    @validator('voice_config')
    def validate_voice_config(cls, v):
        required_fields = ['voice']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Missing required voice config field: {field}')
        
        # Validate voice selection
        allowed_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if v['voice'] not in allowed_voices:
            raise ValueError(f'Invalid voice. Allowed: {allowed_voices}')
        
        # Validate speed if provided
        if 'speed' in v and not (0.25 <= v['speed'] <= 4.0):
            raise ValueError('Speed must be between 0.25 and 4.0')
            
        return v

# --------- Enhanced Voice Configuration Functions ---------

def get_character_voice_config(character_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate voice configuration based on character persona and appearance"""
    try:
        persona = character_data.get('persona', {})
        appearance = character_data.get('appearance', {})
        
        # Check if voice config is already saved in persona
        if 'voice_config' in persona and persona['voice_config']:
            saved_config = persona['voice_config']
            # Validate saved config
            if 'voice' in saved_config and saved_config['voice'] in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
                return {
                    "voice": saved_config.get('voice', 'alloy'),
                    "speed": saved_config.get('speed', 1.0),
                    "model": "tts-1-hd"
                }
        
        # Generate default config based on character traits
        gender = appearance.get('gender', 'person').lower()
        style = persona.get('style', '').lower()
        age = str(appearance.get('age', '25'))
        
        # Voice selection logic with better defaults
        if any(term in gender for term in ['woman', 'female', 'girl']):
            if any(term in style for term in ['elegant', 'formal', 'professional']):
                voice = "nova"
            elif any(term in style for term in ['playful', 'energetic', 'cheerful', 'bubbly']):
                voice = "shimmer"
            else:
                voice = "alloy"
        elif any(term in gender for term in ['man', 'male', 'boy']):
            if any(term in style for term in ['deep', 'serious', 'authoritative', 'confident']):
                voice = "onyx"
            elif any(term in style for term in ['calm', 'wise', 'thoughtful', 'gentle']):
                voice = "echo"
            else:
                voice = "fable"
        else:
            voice = "alloy"  # Default neutral
        
        # Speed based on personality with more nuanced logic
        speed = 1.0
        if any(term in style for term in ['energetic', 'excited', 'fast', 'quick']):
            speed = 1.15
        elif any(term in style for term in ['calm', 'relaxed', 'slow', 'deliberate']):
            speed = 0.9
        elif any(term in style for term in ['very energetic', 'hyperactive']):
            speed = 1.25
        
        # Age adjustments
        try:
            age_num = int(age)
            if age_num < 20:
                speed = min(speed + 0.1, 1.3)  # Slightly faster for younger characters
            elif age_num > 60:
                speed = max(speed - 0.1, 0.8)  # Slightly slower for older characters
        except ValueError:
            pass  # Invalid age, use default speed
        
        return {
            "voice": voice,
            "speed": round(speed, 2),
            "model": "tts-1-hd"
        }
        
    except Exception as e:
        logger.error(f"Error generating voice config: {str(e)}")
        # Return safe defaults
        return {
            "voice": "alloy",
            "speed": 1.0,
            "model": "tts-1-hd"
        }

# --------- Enhanced Audio Processing Functions ---------

def process_audio_file(audio_bytes: bytes, input_format: str) -> str:
    """Process audio file and return path to processed file"""
    temp_files = []  # Track files for cleanup
    
    try:
        # Create temporary file for input audio
        with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            input_path = temp_file.name
            temp_files.append(input_path)
        
        # Convert to WAV if needed (better compatibility with Whisper)
        if input_format.lower() in ['webm', 'ogg', 'm4a']:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(input_path)
                
                # Convert to WAV with optimal settings for Whisper
                wav_path = input_path.replace(f".{input_format}", ".wav")
                audio.export(wav_path, format="wav", parameters=["-ar", "16000"])  # 16kHz sample rate
                temp_files.append(wav_path)
                return wav_path
            except ImportError:
                logger.error("pydub not available, trying direct file processing")
                return input_path
            except Exception as e:
                logger.warning(f"Audio conversion failed, using original: {str(e)}")
                return input_path
        else:
            return input_path
            
    except Exception as e:
        # Cleanup on error
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        raise e

def cleanup_temp_files(file_paths: List[str]):
    """Safely cleanup temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {str(e)}")

# --------- Enhanced Voice Processing Endpoints ---------

@app.post("/voice/speech-to-text")
async def speech_to_text(req: VoiceRequest):
    """Convert user speech to text using OpenAI Whisper with enhanced error handling"""
    temp_files = []
    
    try:
        logger.info(f"Processing speech-to-text for user {req.user_id}, format: {req.format}")
        
        # Validate audio data size (reasonable limits)
        audio_bytes = base64.b64decode(req.audio_data)
        audio_size_mb = len(audio_bytes) / (1024 * 1024)
        
        if audio_size_mb > 25:  # OpenAI limit is 25MB
            raise HTTPException(status_code=400, detail="Audio file too large (max 25MB)")
        
        if audio_size_mb < 0.01:  # Too small to be meaningful
            raise HTTPException(status_code=400, detail="Audio file too small or empty")
        
        logger.info(f"Audio file size: {audio_size_mb:.2f}MB")
        
        # Process audio file
        processed_file_path = process_audio_file(audio_bytes, req.format)
        temp_files.append(processed_file_path)
        
        # Verify file exists and has content
        if not os.path.exists(processed_file_path) or os.path.getsize(processed_file_path) == 0:
            raise HTTPException(status_code=500, detail="Audio processing failed - empty file")
        
        # Transcribe using OpenAI Whisper
        try:
            with open(processed_file_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # You can make this configurable
                    response_format="json"
                )
            
            transcribed_text = transcript.text.strip()
            logger.info(f"Transcribed text length: {len(transcribed_text)} chars")
            
            if not transcribed_text:
                raise HTTPException(status_code=400, detail="No speech detected in audio")
            
            return {
                "transcribed_text": transcribed_text,
                "confidence": "high",
                "audio_duration_seconds": None,  # Whisper doesn't provide this
                "processing_info": {
                    "original_format": req.format,
                    "file_size_mb": round(audio_size_mb, 2)
                }
            }
            
        except Exception as e:
            if "invalid_request_error" in str(e).lower():
                raise HTTPException(status_code=400, detail="Audio format not supported or corrupted")
            else:
                raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech-to-text error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    
    finally:
        # Always cleanup temporary files
        cleanup_temp_files(temp_files)

@app.post("/voice/text-to-speech")
async def text_to_speech(req: TextToSpeechRequest):
    """Convert AI response to speech with character-specific voice and enhanced error handling"""
    try:
        logger.info(f"Generating speech for character {req.character_id}, text length: {len(req.text)}")
        
        # Get character data for voice configuration
        character_result = supabase.table("characters").select("*").eq("id", req.character_id).execute()
        if not character_result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        
        character = character_result.data[0]
        
        # Get voice configuration with error handling
        try:
            voice_config = get_character_voice_config(character)
            logger.info(f"Voice config for character: {voice_config}")
        except Exception as e:
            logger.warning(f"Could not get character voice config: {str(e)}, using defaults")
            voice_config = {"voice": "alloy", "speed": 1.0, "model": "tts-1-hd"}
        
        # Override voice if specified in request
        if req.voice_style and req.voice_style in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
            voice_config["voice"] = req.voice_style
        
        # Validate voice configuration
        if voice_config["voice"] not in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
            logger.warning(f"Invalid voice {voice_config['voice']}, defaulting to alloy")
            voice_config["voice"] = "alloy"
        
        # Ensure speed is within valid range
        voice_config["speed"] = max(0.25, min(4.0, voice_config.get("speed", 1.0)))
        
        logger.info(f"Final voice config: {voice_config}")
        
        # Generate speech using OpenAI TTS with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = openai_client.audio.speech.create(
                    model=voice_config["model"],
                    voice=voice_config["voice"],
                    input=req.text,
                    speed=voice_config["speed"]
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"TTS attempt {attempt + 1} failed: {str(e)}, retrying...")
                    # Try with default settings on retry
                    voice_config = {"voice": "alloy", "speed": 1.0, "model": "tts-1"}
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Speech generation failed after {max_retries + 1} attempts: {str(e)}")
        
        # Validate response
        if not hasattr(response, 'content') or not response.content:
            raise HTTPException(status_code=500, detail="Empty audio response from TTS service")
        
        # Convert response to base64 for frontend
        audio_data = base64.b64encode(response.content).decode('utf-8')
        
        logger.info(f"Generated audio size: {len(response.content)} bytes")
        
        return {
            "audio_data": audio_data,
            "format": "mp3",
            "voice_used": voice_config["voice"],
            "speed_used": voice_config["speed"],
            "text": req.text,
            "audio_size_bytes": len(response.content)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@app.put("/characters/{character_id}/voice-settings")
async def update_voice_settings(character_id: str, req: VoiceSettingsRequest):
    """Update voice settings for a character with enhanced error handling"""
    try:
        logger.info(f"Updating voice settings for character {character_id}, user {req.user_id}")
        logger.info(f"New voice config: {req.voice_config}")
        
        # Verify character ownership and existence
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", req.user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        current_character = result.data[0]
        logger.info(f"Found character: {current_character.get('name')}")
        
        # Safely update persona with voice configuration
        try:
            current_persona = current_character.get("persona", {})
            if not isinstance(current_persona, dict):
                logger.warning("Invalid persona format, resetting to empty dict")
                current_persona = {}
            
            # Update voice config while preserving other persona data
            current_persona["voice_config"] = req.voice_config
            
            logger.info(f"Updated persona structure: {current_persona}")
            
            # Update character in database with detailed error handling
            update_result = supabase.table("characters").update({
                "persona": current_persona,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", character_id).execute()
            
            logger.info(f"Database update result: {update_result}")
            
            if not update_result.data:
                logger.error("Database update returned empty data")
                # Check if there was an error in the response
                if hasattr(update_result, 'error') and update_result.error:
                    raise HTTPException(status_code=500, detail=f"Database error: {update_result.error}")
                else:
                    raise HTTPException(status_code=500, detail="Failed to update character - no data returned")
            
            updated_character = update_result.data[0]
            logger.info("Voice settings updated successfully")
            
            return {
                "character": updated_character,
                "message": "Voice settings updated successfully",
                "voice_config": req.voice_config
            }
            
        except Exception as db_error:
            logger.error(f"Database operation error: {str(db_error)}")
            logger.error(f"Error type: {type(db_error).__name__}")
            raise HTTPException(status_code=500, detail=f"Database update failed: {str(db_error)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice settings update error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice settings update failed: {str(e)}")

@app.get("/characters/{character_id}/voice-settings")
async def get_voice_settings(character_id: str, user_id: str = Query(...)):
    """Get current voice settings for a character"""
    try:
        # Verify character ownership
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        character = result.data[0]
        voice_config = get_character_voice_config(character)
        
        return {
            "character_id": character_id,
            "voice_config": voice_config,
            "character_name": character.get("name")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get voice settings error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get voice settings")

@app.get("/voice/available-voices")
async def get_available_voices():
    """Get list of available voices with detailed descriptions"""
    return {
        "voices": [
            {
                "id": "alloy", 
                "name": "Alloy", 
                "description": "Neutral, versatile voice suitable for any character",
                "gender": "neutral",
                "characteristics": ["balanced", "clear", "professional"],
                "suitable_for": ["general", "professional", "friendly"]
            },
            {
                "id": "echo", 
                "name": "Echo", 
                "description": "Calm, wise male voice with thoughtful delivery",
                "gender": "male",
                "characteristics": ["calm", "deep", "thoughtful"],
                "suitable_for": ["mentor", "wise", "gentle", "older characters"]
            },
            {
                "id": "fable", 
                "name": "Fable", 
                "description": "Warm, approachable male voice",
                "gender": "male",
                "characteristics": ["warm", "friendly", "approachable"],
                "suitable_for": ["storytelling", "friendly", "casual", "young adult"]
            },
            {
                "id": "onyx", 
                "name": "Onyx", 
                "description": "Deep, authoritative male voice",
                "gender": "male",
                "characteristics": ["deep", "authoritative", "confident"],
                "suitable_for": ["serious", "professional", "confident", "leader"]
            },
            {
                "id": "nova", 
                "name": "Nova", 
                "description": "Professional, elegant female voice",
                "gender": "female",
                "characteristics": ["professional", "clear", "elegant"],
                "suitable_for": ["professional", "elegant", "sophisticated", "mature"]
            },
            {
                "id": "shimmer", 
                "name": "Shimmer", 
                "description": "Bright, energetic female voice",
                "gender": "female",
                "characteristics": ["bright", "energetic", "cheerful"],
                "suitable_for": ["playful", "energetic", "cheerful", "young"]
            }
        ],
        "speed_range": {
            "min": 0.25,
            "max": 4.0,
            "default": 1.0,
            "recommendations": {
                "energetic": 1.15,
                "normal": 1.0,
                "calm": 0.9,
                "very_energetic": 1.25,
                "very_calm": 0.8
            }
        }
    }

@app.post("/voice/chat")
async def voice_chat(req: VoiceRequest):
    """Complete voice chat flow with comprehensive error handling and timeout management"""
    temp_files = []
    
    try:
        logger.info(f"Voice chat request for user {req.user_id}, character {req.character_id}")
        
        # Step 1: Convert speech to text with timeout
        try:
            stt_response = await speech_to_text(req)
            user_message = stt_response["transcribed_text"]
            
            if not user_message.strip():
                raise HTTPException(status_code=400, detail="No speech detected in audio")
                
            logger.info(f"Transcribed message: {user_message[:100]}...")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")
        
        # Step 2: Generate AI response (reuse existing chat logic with voice optimizations)
        try:
            character_result = supabase.table("characters").select("*").eq("id", req.character_id).execute()
            if not character_result.data:
                raise HTTPException(status_code=404, detail="Character not found")
            
            character = character_result.data[0]
            persona = character.get("persona", {})
            
            # Build voice-optimized system prompt
            character_name = persona.get('name', character.get('name', 'AI Companion'))
            character_style = persona.get('style', 'friendly and supportive')
            character_bio = persona.get('bio', '')
            
            system_prompt = f"""You are {character_name}, an AI companion in a voice conversation.
            Your personality style: {character_style}
            {f"Background: {character_bio}" if character_bio else ""}
            
            IMPORTANT - Voice Chat Guidelines:
            - Keep responses conversational and natural for speech (60-150 words ideal)
            - Use shorter sentences that flow well when spoken aloud
            - Avoid complex formatting, lists, or special characters
            - Be engaging and expressive in tone
            - Use contractions and natural speech patterns
            - Don't mention images or visual elements in voice mode
            - Respond as if you're having a natural spoken conversation
            """

            # Get recent conversation context (fewer for voice to keep it snappy)
            try:
                recent_memories = supabase.table("memories").select("message, response")\
                    .eq("user_id", req.user_id)\
                    .eq("character_id", req.character_id)\
                    .order("created_at", desc=True)\
                    .limit(3)\
                    .execute()
                
                conversation_history = []
                if recent_memories.data:
                    for memory in reversed(recent_memories.data):
                        if memory.get("message"):
                            conversation_history.append({"role": "user", "content": memory["message"]})
                        if memory.get("response"):
                            conversation_history.append({"role": "assistant", "content": memory["response"]})
            
            except Exception as e:
                logger.warning(f"Could not load conversation history: {str(e)}")
                conversation_history = []

            # Generate AI response with voice-optimized settings
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": user_message})

            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,  # Shorter for voice conversations
                temperature=0.8,
                presence_penalty=0.1  # Encourage varied responses
            )
            ai_response = completion.choices[0].message.content.strip()
            logger.info(f"AI response length: {len(ai_response)} chars")
            
        except Exception as e:
            logger.error(f"AI response generation error: {str(e)}")
            ai_response = "I'm having trouble understanding right now. Could you try again?"
        
        # Step 3: Convert AI response to speech
        try:
            tts_request = TextToSpeechRequest(
                text=ai_response,
                character_id=req.character_id,
                voice_style=req.voice_style if hasattr(req, 'voice_style') else None
            )
            tts_response = await text_to_speech(tts_request)
            
        except Exception as e:
            logger.error(f"TTS generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not generate speech: {str(e)}")
        
        # Step 4: Save conversation to memories
        try:
            memory_data = {
                "user_id": req.user_id,
                "character_id": req.character_id,
                "message": user_message,
                "response": ai_response,
                "interaction_type": "voice",
                "created_at": datetime.utcnow().isoformat()
            }
            memory_result = supabase.table("memories").insert(memory_data).execute()
            logger.info("Voice conversation saved to memories")
            
        except Exception as e:
            logger.warning(f"Could not save voice conversation: {str(e)}")
        
        return {
            "transcribed_text": user_message,
            "ai_response": ai_response,
            "audio_data": tts_response["audio_data"],
            "audio_format": tts_response["format"],
            "voice_used": tts_response["voice_used"],
            "processing_info": {
                "transcription_success": True,
                "tts_success": True,
                "conversation_saved": True
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")
    
    finally:
        # Cleanup any temporary files
        cleanup_temp_files(temp_files)

# --------- Voice Debugging and Testing Endpoints ---------

@app.post("/voice/test-tts")
async def test_text_to_speech(
    text: str = Query(default="Hello! This is a test of the text-to-speech system."), 
    voice: str = Query(default="alloy")
):
    """Test text-to-speech functionality with detailed debugging"""
    try:
        logger.info(f"Testing TTS with voice: {voice}, text: {text[:50]}...")
        
        # Validate inputs
        if voice not in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
            raise HTTPException(status_code=400, detail=f"Invalid voice. Use: alloy, echo, fable, onyx, nova, shimmer")
        
        if len(text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long for test (max 1000 chars)")
        
        # Test TTS
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.0
        )
        
        audio_data = base64.b64encode(response.content).decode('utf-8')
        
        return {
            "audio_data": audio_data,
            "format": "mp3",
            "text": text,
            "voice": voice,
            "test_success": True,
            "audio_size_bytes": len(response.content)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS test failed: {str(e)}")

@app.post("/voice/debug-audio")
async def debug_audio_upload(req: VoiceRequest):
    """Debug audio upload issues"""
    temp_files = []
    
    try:
        logger.info(f"Debug audio upload - User: {req.user_id}, Format: {req.format}")
        
        # Decode and analyze audio data
        try:
            audio_bytes = base64.b64decode(req.audio_data)
            audio_size = len(audio_bytes)
            logger.info(f"Audio decoded successfully - Size: {audio_size} bytes")
        except Exception as e:
            return {
                "error": f"Audio processing failed: {str(e)}",
                "success": False,
                "details": str(e)
            }
                
    except Exception as e:
        logger.error(f"Debug audio error: {str(e)}")
        return {
            "error": "Debug process failed",
            "details": str(e),
            "success": False
        }
    
    finally:
        cleanup_temp_files(temp_files)

# --------- Enhanced Voice Analytics ---------

@app.get("/users/{user_id}/voice-stats")
async def get_voice_stats(user_id: str):
    """Get comprehensive voice interaction statistics"""
    try:
        # Get voice interaction count
        voice_interactions = supabase.table("memories").select("id, created_at")\
            .eq("user_id", user_id)\
            .eq("interaction_type", "voice")\
            .execute()
        
        voice_count = len(voice_interactions.data) if voice_interactions.data else 0
        
        # Get total interactions for comparison
        total_interactions = supabase.table("memories").select("id")\
            .eq("user_id", user_id)\
            .execute()
        
        total_count = len(total_interactions.data) if total_interactions.data else 0
        
        # Calculate recent activity (last 7 days)
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent_voice = supabase.table("memories").select("id")\
            .eq("user_id", user_id)\
            .eq("interaction_type", "voice")\
            .gte("created_at", seven_days_ago)\
            .execute()
        
        recent_voice_count = len(recent_voice.data) if recent_voice.data else 0
        
        return {
            "voice_interactions": voice_count,
            "total_interactions": total_count,
            "voice_percentage": round((voice_count / total_count * 100) if total_count > 0 else 0, 1),
            "recent_voice_interactions": recent_voice_count,
            "voice_enabled": True
        }
    
    except Exception as e:
        logger.error(f"Voice stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get voice stats")

# --------- Voice Health Check ---------

@app.get("/voice/health")
async def voice_health_check():
    """Check if voice services are working properly"""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "overall_status": "healthy"
    }
    
    # Test OpenAI TTS
    try:
        test_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Health check test",
            speed=1.0
        )
        health_status["services"]["tts"] = {
            "status": "healthy",
            "response_size": len(test_response.content)
        }
    except Exception as e:
        health_status["services"]["tts"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    # Test audio processing capabilities
    try:
        # Test if pydub is available
        from pydub import AudioSegment
        health_status["services"]["audio_processing"] = {
            "status": "healthy",
            "pydub_available": True
        }
    except ImportError:
        health_status["services"]["audio_processing"] = {
            "status": "degraded",
            "pydub_available": False,
            "warning": "Audio format conversion may be limited"
        }
    except Exception as e:
        health_status["services"]["audio_processing"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    # Test database connectivity for voice features
    try:
        supabase.table("characters").select("id").limit(1).execute()
        health_status["services"]["database"] = {
            "status": "healthy"
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "unhealthy"
    
    return health_status

# --------- Voice Error Recovery ---------

@app.post("/voice/recovery/reset-session")
async def reset_voice_session(user_id: str = Query(...), character_id: str = Query(...)):
    """Reset voice session state (useful for stuck recordings)"""
    try:
        logger.info(f"Resetting voice session for user {user_id}, character {character_id}")
        
        # This endpoint doesn't need to do much on the backend since voice state 
        # is primarily frontend-managed, but we can clear any temporary data
        
        # Could add logic here to:
        # - Clear any pending voice operations
        # - Reset voice-related session data
        # - Clean up any stuck temporary files
        
        return {
            "message": "Voice session reset successfully",
            "user_id": user_id,
            "character_id": character_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Voice session reset error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reset voice session")

# --------- Voice Configuration Validation ---------

@app.post("/voice/validate-config")
async def validate_voice_config(config: Dict[str, Any]):
    """Validate voice configuration before saving"""
    try:
        errors = []
        warnings = []
        
        # Check required fields
        if 'voice' not in config:
            errors.append("Missing 'voice' field")
        elif config['voice'] not in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
            errors.append(f"Invalid voice '{config['voice']}'. Must be one of: alloy, echo, fable, onyx, nova, shimmer")
        
        # Check optional fields
        if 'speed' in config:
            speed = config['speed']
            if not isinstance(speed, (int, float)):
                errors.append("Speed must be a number")
            elif not (0.25 <= speed <= 4.0):
                errors.append("Speed must be between 0.25 and 4.0")
            elif speed > 1.5:
                warnings.append("Very high speed may affect speech clarity")
            elif speed < 0.75:
                warnings.append("Very low speed may sound unnatural")
        
        # Check for unknown fields
        known_fields = ['voice', 'speed', 'model']
        unknown_fields = [field for field in config.keys() if field not in known_fields]
        if unknown_fields:
            warnings.append(f"Unknown fields will be ignored: {unknown_fields}")
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "validated_config": {
                "voice": config.get('voice', 'alloy'),
                "speed": config.get('speed', 1.0),
                "model": "tts-1-hd"
            } if is_valid else None
        }
        
    except Exception as e:
        logger.error(f"Voice config validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation failed")

# --------- Additional Utility Endpoints ---------

@app.get("/voice/character-voices")
async def get_character_voice_recommendations():
    """Get voice recommendations for different character types"""
    return {
        "recommendations": {
            "professional_woman": {
                "voice": "nova",
                "speed": 1.0,
                "description": "Clear, professional female voice"
            },
            "energetic_woman": {
                "voice": "shimmer", 
                "speed": 1.1,
                "description": "Bright, cheerful female voice"
            },
            "authoritative_man": {
                "voice": "onyx",
                "speed": 0.95,
                "description": "Deep, confident male voice"
            },
            "friendly_man": {
                "voice": "fable",
                "speed": 1.0,
                "description": "Warm, approachable male voice"
            },
            "wise_mentor": {
                "voice": "echo",
                "speed": 0.9,
                "description": "Calm, thoughtful male voice"
            },
            "neutral_assistant": {
                "voice": "alloy",
                "speed": 1.0,
                "description": "Balanced, versatile voice"
            }
        },
        "customization_tips": [
            "Match voice gender to character appearance when possible",
            "Adjust speed based on character energy level",
            "Professional characters work well with moderate speeds (0.9-1.1)",
            "Energetic characters can use faster speeds (1.1-1.3)",
            "Calm characters benefit from slower speeds (0.8-0.95)"
        ]
    }

# --------- Error Recovery Functions ---------

def safe_persona_update(current_persona: Any, voice_config: Dict[str, Any]) -> Dict[str, Any]:
    """Safely update persona with voice config, handling various data structures"""
    try:
        # Handle case where persona might not be a dict
        if not isinstance(current_persona, dict):
            logger.warning(f"Persona is not a dict (type: {type(current_persona)}), creating new dict")
            persona = {}
        else:
            persona = current_persona.copy()
        
        # Ensure voice_config is properly structured
        validated_voice_config = {
            "voice": voice_config.get("voice", "alloy"),
            "speed": float(voice_config.get("speed", 1.0)),
            "model": voice_config.get("model", "tts-1-hd")
        }
        
        persona["voice_config"] = validated_voice_config
        return persona
        
    except Exception as e:
        logger.error(f"Error in safe_persona_update: {str(e)}")
        # Return minimal valid structure
        return {
            "voice_config": {
                "voice": "alloy",
                "speed": 1.0,
                "model": "tts-1-hd"
            }
        }

# Update the existing update_voice_settings function to use safe_persona_update
# (This replaces the previous version)

@app.put("/characters/{character_id}/voice-settings-v2")
async def update_voice_settings_enhanced(character_id: str, req: VoiceSettingsRequest):
    """Enhanced voice settings update with comprehensive error handling"""
    try:
        logger.info(f"Enhanced voice settings update - Character: {character_id}, User: {req.user_id}")
        logger.info(f"Voice config to save: {req.voice_config}")
        
        # Step 1: Verify character ownership
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", req.user_id).execute()
        if not result.data:
            logger.warning(f"Character {character_id} not found for user {req.user_id}")
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        current_character = result.data[0]
        logger.info(f"Found character: {current_character.get('name')}")
        
        # Step 2: Validate voice configuration
        validation_response = await validate_voice_config(req.voice_config)
        if not validation_response["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid voice configuration: {validation_response['errors']}")
        
        validated_config = validation_response["validated_config"]
        logger.info(f"Validated voice config: {validated_config}")
        
        # Step 3: Safely update persona
        current_persona = current_character.get("persona", {})
        logger.info(f"Current persona type: {type(current_persona)}")
        
        updated_persona = safe_persona_update(current_persona, validated_config)
        logger.info(f"Updated persona structure created successfully")
        
        # Step 4: Update database with transaction-like approach
        try:
            # Prepare update data
            update_data = {
                "persona": updated_persona,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Perform update
            update_result = supabase.table("characters").update(update_data).eq("id", character_id).eq("user_id", req.user_id).execute()
            
            logger.info(f"Database update completed. Result type: {type(update_result)}")
            
            # Verify update succeeded
            if not update_result.data:
                logger.error("Database update returned no data")
                # Try to fetch the character again to see current state
                verification = supabase.table("characters").select("*").eq("id", character_id).execute()
                logger.info(f"Verification fetch result: {verification.data is not None}")
                
                raise HTTPException(status_code=500, detail="Update operation failed - no data returned from database")
            
            updated_character = update_result.data[0]
            logger.info("Voice settings saved successfully to database")
            
            # Step 5: Verify the save worked
            saved_voice_config = updated_character.get("persona", {}).get("voice_config")
            if not saved_voice_config:
                logger.warning("Voice config not found in saved character data")
            else:
                logger.info(f"Verified saved voice config: {saved_voice_config}")
            
            return {
                "success": True,
                "character": updated_character,
                "message": "Voice settings updated successfully",
                "voice_config": validated_config,
                "warnings": validation_response.get("warnings", [])
            }
            
        except Exception as db_error:
            logger.error(f"Database operation failed: {type(db_error).__name__}: {str(db_error)}")
            
            # Provide more specific error messages
            error_msg = str(db_error).lower()
            if "constraint" in error_msg:
                raise HTTPException(status_code=400, detail="Voice settings violate database constraints")
            elif "permission" in error_msg or "auth" in error_msg:
                raise HTTPException(status_code=403, detail="Permission denied for voice settings update")
            elif "timeout" in error_msg:
                raise HTTPException(status_code=408, detail="Database timeout - please try again")
            else:
                raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice settings update failed: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice settings update failed: {str(e)}")

# --------- Voice Session Management ---------

@app.post("/voice/start-session")
async def start_voice_session(user_id: str = Query(...), character_id: str = Query(...)):
    """Initialize a voice chat session with proper setup"""
    try:
        # Verify character exists and get voice config
        character_result = supabase.table("characters").select("*").eq("id", character_id).execute()
        if not character_result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        
        character = character_result.data[0]
        voice_config = get_character_voice_config(character)
        
        # Return session info
        return {
            "session_id": f"{user_id}_{character_id}_{int(datetime.utcnow().timestamp())}",
            "character": {
                "id": character_id,
                "name": character.get("name"),
                "voice_config": voice_config
            },
            "session_settings": {
                "max_recording_duration": 30,  # seconds
                "auto_stop_silence": 3,  # seconds of silence
                "audio_format": "webm"
            },
            "status": "ready"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice session start error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start voice session")

@app.post("/voice/end-session")
async def end_voice_session(session_id: str = Query(...)):
    """Clean up voice session"""
    try:
        logger.info(f"Ending voice session: {session_id}")
        
        # Add any cleanup logic here
        # - Clear temporary files
        # - Reset any session state
        # - Log session statistics
        
        return {
            "message": "Voice session ended successfully",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error ending voice session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to end voice session")

# --------- Frontend Helper Endpoints ---------

@app.get("/voice/browser-support")
async def check_browser_voice_support():
    """Provide information about browser voice support requirements"""
    return {
        "required_features": [
            "MediaRecorder API",
            "getUserMedia API", 
            "Audio playback support",
            "Base64 encoding/decoding"
        ],
        "supported_formats": {
            "input": ["webm", "ogg", "wav", "mp3", "m4a"],
            "output": ["mp3"]
        },
        "browser_requirements": {
            "chrome": "≥ 60",
            "firefox": "≥ 65", 
            "safari": "≥ 14",
            "edge": "≥ 79"
        },
        "troubleshooting": {
            "infinite_recording": [
                "Check microphone permissions",
                "Try refreshing the page",
                "Check browser console for errors",
                "Use /voice/recovery/reset-session endpoint"
            ],
            "no_audio_output": [
                "Check device volume settings",
                "Verify browser audio permissions",
                "Test with /voice/test-tts endpoint"
            ],
            "settings_not_saving": [
                "Verify voice configuration format",
                "Check network connection",
                "Use /voice/validate-config to test configuration"
            ]
        }
    }

# --------- Installation and Dependency Check ---------

@app.get("/voice/system-info")
async def get_voice_system_info():
    """Get information about voice processing capabilities"""
    system_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {},
        "capabilities": {}
    }
    
    # Check pydub
    try:
        import pydub
        system_info["dependencies"]["pydub"] = {
            "available": True,
            "version": getattr(pydub, '__version__', 'unknown')
        }
        system_info["capabilities"]["audio_conversion"] = True
    except ImportError:
        system_info["dependencies"]["pydub"] = {
            "available": False,
            "error": "pydub not installed"
        }
        system_info["capabilities"]["audio_conversion"] = False
    
    # Check ffmpeg (indirectly through pydub)
    try:
        from pydub.utils import which
        ffmpeg_path = which("ffmpeg")
        system_info["dependencies"]["ffmpeg"] = {
            "available": ffmpeg_path is not None,
            "path": ffmpeg_path
        }
    except:
        system_info["dependencies"]["ffmpeg"] = {
            "available": False,
            "error": "Cannot detect ffmpeg"
        }
    
    # Check OpenAI client
    try:
        # Test if we can create a client (doesn't make actual API call)
        test_client = OpenAI(api_key="test")
        system_info["dependencies"]["openai"] = {
            "available": True,
            "client_created": True
        }
        system_info["capabilities"]["speech_services"] = True
    except Exception as e:
        system_info["dependencies"]["openai"] = {
            "available": False,
            "error": str(e)
        }
        system_info["capabilities"]["speech_services"] = False
    
    return system_info {
                "error": "Invalid base64 audio data",
                "details": str(e),
                "success": False
            }
        
        # Try to process the audio file
        try:
            processed_path = process_audio_file(audio_bytes, req.format)
            temp_files.append(processed_path)
            
            if os.path.exists(processed_path):
                processed_size = os.path.getsize(processed_path)
                logger.info(f"Audio processed successfully - Size: {processed_size} bytes")
                
                return {
                    "success": True,
                    "original_size_bytes": audio_size,
                    "processed_size_bytes": processed_size,
                    "format": req.format,
                    "processing_successful": True,
                    "message": "Audio processing completed successfully"
                }
            else:
                return {
                    "error": "Processed file not found",
                    "success": False
                }
                
        except Exception as e:
            return                     

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
