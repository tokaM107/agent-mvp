import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "transport_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

def check_database():
    try:
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT, 
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        
        print(f"\n--- Database Status Report ({DB_NAME}) ---\n")

        # 1. Check the main tables
        tables = ["stop", "route", "trip", "route_stop", "agency"]
        total_rows = 0
        
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
                total_rows += count
                status = "Filled" if count > 0 else "Empty"
                print(f"Table {table.ljust(12)}: {str(count).ljust(6)} rows ({status})")
            except Exception as e:
                print(f"Table {table.ljust(12)}: Does not exist! (Error: {e})")

        print("-" * 40)

        # 2. Check extensions (PostGIS & pgRouting)
        extensions = ["postgis", "pgrouting", "pg_trgm"]
        print("\n--- Extensions ---")
        for ext in extensions:
            cur.execute(f"SELECT count(*) FROM pg_extension WHERE extname = '{ext}';")
            exists = cur.fetchone()[0] > 0
            status = "Enabled" if exists else "Not enabled"
            print(f"{ext.ljust(15)}: {status}")

        print("\n" + "="*40)
        
        if total_rows == 0:
            print("\nThe database is completely empty! You need to run the import script.")
        else:
            print("\nThe database is ready and working. You can continue confidently.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"\nFailed to connect to the database: {e}")
        print("Make sure the Docker container is running and the password is correct.")

if __name__ == "__main__":
    check_database()
