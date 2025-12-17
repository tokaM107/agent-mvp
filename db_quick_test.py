import os
from tools import search_stop_by_name_db, find_journeys_db, get_db_connection

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_NAME", "transport_db")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "postgres")

conn = get_db_connection()
print("DB conn:", bool(conn))
if conn:
    conn.close()

stops = search_stop_by_name_db("Raml", 5)
print("Fuzzy stops:", stops)

if len(stops) >= 2:
    o = stops[0]["stop_id"]
    d = stops[1]["stop_id"]
    journeys = find_journeys_db(o, d, max_results=5)
    print("Journeys:", journeys)
else:
    print("Not enough stops to test journeys.")
