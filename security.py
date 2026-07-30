import os
import time
from passlib.context import CryptContext
from itsdangerous import Signer, BadSignature
import bcrypt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "ecosystem_ultra_secure_sentinel_token_key_2026")
signer = Signer(SECRET_KEY)


# Session configuration: 2 hours lifetime (in seconds)
SESSION_LIFETIME_SECONDS = 7200 

def hash_password(password: str) -> str:
    """Encodes a clean string password to bytes, hashes it, and saves it as a string."""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed_bytes = bcrypt.hashpw(password_bytes, salt)
    return hashed_bytes.decode('utf-8')  # Saved safely to your VARCHAR column

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies text entry cleanly against your stored PostgreSQL data string."""
    try:
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False
def create_session_token(user_id: int) -> str:
    """Creates a token containing the user ID and a creation timestamp."""
    current_time = int(time.time())
    payload = f"{user_id}:{current_time}"
    return signer.sign(payload.encode()).decode()

def verify_session_token(token: str) -> int | None:
    """Verifies signature structure and checks if token has expired past its TTL."""
    try:
        unsigned_bytes = signer.unsign(token.encode())
        payload = unsigned_bytes.decode()
        
        user_id_str, timestamp_str = payload.split(":")
        user_id = int(user_id_str)
        timestamp = int(timestamp_str)
        
        # Check if the session has expired
        if int(time.time()) - timestamp > SESSION_LIFETIME_SECONDS:
            return None # Expired session
            
        return user_id
    except (BadSignature, ValueError):
        return None