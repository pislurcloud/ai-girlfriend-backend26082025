# Add these endpoints to your main.py file

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
