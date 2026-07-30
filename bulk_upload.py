import os
import io
import csv
import psycopg2
import pandas as pd

class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:hkbk123@localhost:5432/ecosystem_db"
    )

settings = Settings()

def main():
    excel_file = "your_3000_mental_health_matrix.csv"
    print(f"Processing row updates from {excel_file}...")
    
    try:
        df = pd.read_csv(excel_file)
    except FileNotFoundError:
        print(f"[ERROR] '{excel_file}' missing.")
        return

    # Convert table objects to clean string elements
    df['domain'] = df['domain'].astype(str).str.strip()
    df['keyword_combinations'] = df['keyword_combinations'].astype(str).str.strip()
    df['predefined_title'] = df['predefined_title'].astype(str).str.strip()
    df['predefined_answer'] = df['predefined_answer'].astype(str).str.strip()
    df['action_url'] = df['action_url'].fillna('').astype(str).str.strip()
    df['severity_tier'] = df['severity_tier'].fillna(1).astype(int)
    
    df = df[['domain', 'keyword_combinations', 'predefined_title', 'predefined_answer', 'severity_tier', 'action_url']]
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False, sep='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    csv_buffer.seek(0)
    
    conn = None
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()
        
        print("Truncating obsolete rows...")
        cur.execute("TRUNCATE TABLE clinical_knowledge_base RESTART IDENTITY;")
        
        print(f"Uploading {len(df)} conversational text vector blocks...")
        cur.copy_expert(
            "COPY clinical_knowledge_base (domain, keyword_combinations, predefined_title, predefined_answer, severity_tier, action_url) "
            "FROM STDIN WITH (FORMAT csv, DELIMITER '|', QUOTE '\"')", 
            csv_buffer
        )
        conn.commit()
        print("[SUCCESS] The database rows have been permanently updated.")
    except Exception as error:
        print(f"[CRITICAL UPLOAD ERROR] {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    main()