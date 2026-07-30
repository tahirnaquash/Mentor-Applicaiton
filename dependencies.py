from fastapi import Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from database import get_db
import models
from security import verify_session_token

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Enforces active login. Bypasses static paths back to the gateway."""
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        raise HTTPException(status_code=303, detail="No active session found")
        
    user_id = verify_session_token(session_token)
    if not user_id:
        raise HTTPException(status_code=303, detail="Session tampered or expired")
        
    user = db.query(models.DbUser).filter(models.DbUser.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=303, detail="Account suspended or missing")
        
    return user

def enforce_role(required_role: str):
    """Enforces strict individual account boundaries per view."""
    def dependency(current_user: models.DbUser = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(status_code=303, detail="Unauthorized access layer")
        return current_user
    return dependency