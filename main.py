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

# --------- Voice-Related Models ---------

class VoiceRequest(BaseModel):
    user_id: str
    character_id: str
    audio_data: str  # Base64 encoded audio
    format: str = "webm"  # Audio format from browser

class TextToSpeechRequest(BaseModel):
    text: str
    character_id: str
    voice_style: Optional[str] = None

class VoiceSettingsRequest(BaseModel):
    character_id: str
    user_id: str
    voice_config: Dict[str, Any]

# --------- Voice Configuration Functions ---------

def get_character_voice_config(character_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate voice configuration based on character persona and appearance"""
    persona = character_data.get('persona', {})
    appearance = character_data.get('appearance', {})
    
    # Map character traits to OpenAI voice options
    # Available voices: alloy, echo, fable, onyx, nova, shimmer
    
    gender = appearance.get('gender', 'person').lower()
    style = persona.get('style', '').lower()
    age = appearance.get('age', '25')
    
    # Voice selection logic
    if 'woman' in gender or 'female' in gender:
        if 'elegant' in style or 'formal' in style:
            voice = "nova"  # Professional female voice
        elif 'playful' in style or 'energetic' in style:
            voice = "shimmer"  # Bright, energetic female voice
        else:
            voice = "alloy"  # Neutral female voice
    elif 'man' in gender or 'male' in gender:
        if 'deep' in style or 'serious' in style:
            voice = "onyx"  # Deep male voice
        elif 'calm' in style or 'wise' in style:
            voice = "echo"  # Calm male voice
        else:
            voice = "fable"  # Neutral male voice
    else:
        # Non-binary or unspecified
        voice = "alloy"  # Most neutral option
    
    # Speed based on personality
    if 'energetic' in style or 'excited' in style:
        speed = 1.1
    elif 'calm' in style or 'relaxed' in style:
        speed = 0.9
    else:
        speed = 1.0
    
    return {
        "voice": voice,
        "speed": speed,
        "model": "tts-1-hd"  # High quality model
    }

# --------- Voice Processing Endpoints ---------

@app.post("/voice/speech-to-text")
async def speech_to_text(req: VoiceRequest):
    """Convert user speech to text using OpenAI Whisper"""
    try:
        print(f"Processing speech-to-text for user {req.user_id}")
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(req.audio_data)
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=f".{req.format}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Convert to compatible format if needed
            if req.format.lower() in ['webm', 'ogg']:
                # Convert webm/ogg to wav for better compatibility
                audio = AudioSegment.from_file(temp_file_path)
                wav_path = temp_file_path.replace(f".{req.format}", ".wav")
                audio.export(wav_path, format="wav")
                audio_file_path = wav_path
            else:
                audio_file_path = temp_file_path
            
            # Transcribe using OpenAI Whisper
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"  # Auto-detect or specify language
                )
            
            transcribed_text = transcript.text
            print(f"Transcribed text: {transcribed_text}")
            
            # Clean up temporary files
            os.unlink(temp_file_path)
            if audio_file_path != temp_file_path:
                os.unlink(audio_file_path)
            
            return {
                "transcribed_text": transcribed_text,
                "confidence": "high"  # Whisper doesn't provide confidence scores
            }
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
    
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")

@app.post("/voice/text-to-speech")
async def text_to_speech(req: TextToSpeechRequest):
    """Convert AI response to speech with character-specific voice"""
    try:
        print(f"Generating speech for character {req.character_id}")
        
        # Get character data for voice configuration
        character_result = supabase.table("characters").select("*").eq("id", req.character_id).execute()
        if not character_result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        
        character = character_result.data[0]
        voice_config = get_character_voice_config(character)
        
        # Override voice if specified
        if req.voice_style:
            voice_config["voice"] = req.voice_style
        
        print(f"Using voice config: {voice_config}")
        
        # Generate speech using OpenAI TTS
        response = openai_client.audio.speech.create(
            model=voice_config["model"],
            voice=voice_config["voice"],
            input=req.text,
            speed=voice_config["speed"]
        )
        
        # Convert response to base64 for frontend
        audio_data = base64.b64encode(response.content).decode('utf-8')
        
        return {
            "audio_data": audio_data,
            "format": "mp3",
            "voice_used": voice_config["voice"],
            "text": req.text
        }
    
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@app.put("/characters/{character_id}/voice-settings")
async def update_voice_settings(character_id: str, req: VoiceSettingsRequest):
    """Update voice settings for a character"""
    try:
        # Verify character ownership
        result = supabase.table("characters").select("*").eq("id", character_id).eq("user_id", req.user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Character not found or not owned by user")
        
        current_character = result.data[0]
        
        # Update voice configuration
        current_persona = current_character.get("persona", {})
        current_persona["voice_config"] = req.voice_config
        
        update_result = supabase.table("characters").update({
            "persona": current_persona
        }).eq("id", character_id).execute()
        
        if not update_result.data:
            raise HTTPException(status_code=500, detail="Failed to update voice settings")
        
        return {"character": update_result.data[0], "message": "Voice settings updated"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Voice settings update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Voice settings update failed")

@app.get("/voice/available-voices")
async def get_available_voices():
    """Get list of available voices with descriptions"""
    return {
        "voices": [
            {
                "id": "alloy", 
                "name": "Alloy", 
                "description": "Neutral, versatile voice",
                "gender": "neutral",
                "suitable_for": ["general", "professional", "friendly"]
            },
            {
                "id": "echo", 
                "name": "Echo", 
                "description": "Calm, wise male voice",
                "gender": "male",
                "suitable_for": ["mentor", "calm", "thoughtful"]
            },
            {
                "id": "fable", 
                "name": "Fable", 
                "description": "Warm male voice",
                "gender": "male",
                "suitable_for": ["storytelling", "friendly", "approachable"]
            },
            {
                "id": "onyx", 
                "name": "Onyx", 
                "description": "Deep, authoritative male voice",
                "gender": "male",
                "suitable_for": ["serious", "professional", "confident"]
            },
            {
                "id": "nova", 
                "name": "Nova", 
                "description": "Professional female voice",
                "gender": "female",
                "suitable_for": ["professional", "elegant", "sophisticated"]
            },
            {
                "id": "shimmer", 
                "name": "Shimmer", 
                "description": "Bright, energetic female voice",
                "gender": "female",
                "suitable_for": ["playful", "energetic", "cheerful"]
            }
        ]
    }

# --------- Enhanced Chat with Voice Support ---------

@app.post("/chat/voice")
async def voice_chat(req: VoiceRequest):
    """Complete voice chat flow: speech-to-text, AI response, text-to-speech"""
    try:
        print(f"Voice chat request for user {req.user_id}, character {req.character_id}")
        
        # Step 1: Convert speech to text
        stt_response = await speech_to_text(req)
        user_message = stt_response["transcribed_text"]
        
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="No speech detected")
        
        # Step 2: Generate AI response (reuse existing chat logic)
        chat_request = ChatRequest(
            user_id=req.user_id,
            character_id=req.character_id,
            message=user_message
        )
        
        # Get chat response (from existing chat endpoint logic)
        character_result = supabase.table("characters").select("*").eq("id", req.character_id).execute()
        if not character_result.data:
            raise HTTPException(status_code=404, detail="Character not found")
        
        character = character_result.data[0]
        persona = character.get("persona", {})
        
        # Build system prompt (reuse from chat endpoint)
        character_name = persona.get('name', character.get('name', 'AI Companion'))
        character_style = persona.get('style', 'friendly and supportive')
        character_bio = persona.get('bio', '')
        
        system_prompt = f"""You are {character_name}, an AI companion speaking in voice chat.
        Your personality style: {character_style}
        {f"Background: {character_bio}" if character_bio else ""}
        
        Guidelines for voice responses:
        - Keep responses conversational and natural for speech
        - Use shorter sentences that flow well when spoken
        - Avoid complex formatting or lists
        - Be engaging and expressive in tone
        - Don't mention generating images in voice mode
        """

        # Get recent conversation context
        try:
            recent_memories = supabase.table("memories").select("message, response")\
                .eq("user_id", req.user_id)\
                .eq("character_id", req.character_id)\
                .order("created_at", desc=True)\
                .limit(5)\
                .execute()
            
            conversation_history = []
            if recent_memories.data:
                for memory in reversed(recent_memories.data):
                    if memory.get("message"):
                        conversation_history.append({"role": "user", "content": memory["message"]})
                    if memory.get("response"):
                        conversation_history.append({"role": "assistant", "content": memory["response"]})
        
        except Exception as e:
            print(f"Warning: Could not load conversation history: {str(e)}")
            conversation_history = []

        # Generate AI response
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,  # Shorter for voice
            temperature=0.8
        )
        ai_response = completion.choices[0].message.content
        
        # Step 3: Convert AI response to speech
        tts_request = TextToSpeechRequest(
            text=ai_response,
            character_id=req.character_id
        )
        tts_response = await text_to_speech(tts_request)
        
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
            supabase.table("memories").insert(memory_data).execute()
        except Exception as e:
            print(f"Warning: Could not save voice conversation: {str(e)}")
        
        return {
            "transcribed_text": user_message,
            "ai_response": ai_response,
            "audio_data": tts_response["audio_data"],
            "audio_format": tts_response["format"],
            "voice_used": tts_response["voice_used"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Voice chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")

# --------- Voice Analytics ---------

@app.get("/users/{user_id}/voice-stats")
async def get_voice_stats(user_id: str):
    """Get voice interaction statistics"""
    try:
        # Get voice interaction count
        voice_interactions = supabase.table("memories").select("id")\
            .eq("user_id", user_id)\
            .eq("interaction_type", "voice")\
            .execute()
        
        voice_count = len(voice_interactions.data) if voice_interactions.data else 0
        
        # Get total interactions for comparison
        total_interactions = supabase.table("memories").select("id")\
            .eq("user_id", user_id)\
            .execute()
        
        total_count = len(total_interactions.data) if total_interactions.data else 0
        
        return {
            "voice_interactions": voice_count,
            "total_interactions": total_count,
            "voice_percentage": round((voice_count / total_count * 100) if total_count > 0 else 0, 1)
        }
    
    except Exception as e:
        print(f"Voice stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get voice stats")

# --------- Voice Testing Endpoints ---------

@app.post("/voice/test-tts")
async def test_text_to_speech(text: str = "Hello! This is a test of the text-to-speech system.", voice: str = "alloy"):
    """Test text-to-speech functionality"""
    try:
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
            "voice": voice
        }
    
    except Exception as e:
        print(f"TTS test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS test failed: {str(e)}")

# Add to requirements.txt:
# pydub
# ffmpeg-python

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
