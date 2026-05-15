import sys
sys.path.insert(0, ".")
import psycopg2
from config import settings

def check_counts():
    print("Checking embedding counts in Neon Database...\n")
    try:
        conn = psycopg2.connect(settings.NEON_DB_URL)
        cursor = conn.cursor()
        
        # The tables we ingest into
        tables = ["stripe_docs", "stripe_github_issues", "stripe_stackoverflow"]
        
        total = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                total += count
                print(f"✅ {table}: {count} embeddings")
                
                # If we want to see the specific sources in stripe_docs (docs vs changelog)
                if table == "stripe_docs":
                    try:
                        # Langchain/Supabase uses metadata or cmetadata depending on version, skip for safety to avoid transaction aborts
                        pass
                    except Exception as e:
                        conn.rollback()
                        
            except Exception as e:
                print(f"❌ {table}: Table does not exist or error ({e})")
                conn.rollback()
                
        print(f"\nTotal Embeddings Stored: {total}")
        
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_counts()
