import os

class Settings:
    # Format: postgresql://[user]:[password]@[host]:[port]/[database_name]
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:hkbk123@localhost:5432/ecosystem_db"
    )

settings = Settings()