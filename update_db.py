import models
from database import engine, Base
from sqlalchemy import text

print("Connecting to database and running hot-reload structural update...")

with engine.connect() as connection:
    # 1. Drop the old table configuration structure cleanly
    print("Dropping old system_configurations table structure...")
    connection.execute(text("DROP TABLE IF EXISTS system_configurations CASCADE;"))
    connection.commit()
    # Add missing column to academic_records if it does not exist
    print("Ensuring 'is_transferred_to_counselor' column exists in academic_records...")
    connection.execute(text("ALTER TABLE IF EXISTS academic_records ADD COLUMN IF NOT EXISTS is_transferred_to_counselor BOOLEAN DEFAULT FALSE NOT NULL;"))
    connection.commit()

# 2. Re-create the table using the updated models.py definition containing 'department'
print("Re-building table with the new department mapping columns...")
Base.metadata.create_all(bind=engine)

print("Database columns synchronized successfully! You can now delete this file.")