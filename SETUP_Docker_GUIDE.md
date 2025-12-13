# Transport Database Setup Guide

Complete step-by-step guide to set up and run the PostgreSQL 18 + PostGIS + pgRouting database for Alexandria public transportation.

## Quick Start

### 1. Clone and Start Database

```powershell
# Clone the repository
git clone https://github.com/marwan051/final-project-database
cd final-project-database

# Start containers (first time will build the image)
docker compose up -d --build

# Wait ~10 seconds for containers to start
docker ps -a
```

### 2. Fix Line Endings (Windows Only)

Windows may add CRLF line endings that break shell scripts. Fix them:

```powershell
# Install dos2unix in the container
docker exec -it transport-db bash -c "apt-get update && apt-get install -y dos2unix"

# Convert all scripts to Unix format
docker exec -it transport-db bash -c "dos2unix /docker-entrypoint-initdb.d/*.sh /usr/local/bin/*.sh"
```

### 3. Initialize Database Schema

```powershell
# Run init script to create PostGIS and pgRouting extensions
docker exec -it transport-db bash -c "/docker-entrypoint-initdb.d/01-init-database.sh"

# Load operational schema (stop, route, route_geometry, route_stop tables)
docker exec transport-db bash -c "psql -U postgres -d transport_db -f /docker-entrypoint-initdb.d/02-schema.sql"

# Load GTFS staging schema (gtfs_staging_* tables)
docker exec transport-db bash -c "psql -U postgres -d transport_db -f /docker-entrypoint-initdb.d/03-gtfs-staging-schema.sql"

# Load ETL transformation functions
docker exec transport-db bash -c "psql -U postgres -d transport_db -f /docker-entrypoint-initdb.d/04-gtfs-etl-transform.sql"

# Verify tables were created
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';"
```

Expected output: `count = 15` or more.

### 4. Import GTFS Data

```powershell
# Import GTFS CSVs and run ETL transformation
docker exec transport-db bash -c "gtfs2db.sh --with-etl"
```

Expected output:
- Imported: agency, calendar, routes, stops, shapes, trips, stop_times, feed_info
- ETL completed: ~441 stops, ~104 routes, ~1493 route_stops

### 5. Verify Data

```powershell
# Check operational tables
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM stop;"
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM route;"
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM route_stop;"

# Sample stops with coordinates
docker exec transport-db psql -U postgres -d transport_db -c "SELECT stop_id, name, ST_X(geom_4326) AS lon, ST_Y(geom_4326) AS lat FROM stop LIMIT 5;"

# Sample routes
docker exec transport-db psql -U postgres -d transport_db -c "SELECT route_id, name, mode FROM route LIMIT 5;"
```

### 6. Import OSM Street Network (pgRouting) ✨ NEW

This enables walking distance calculations between stops for transfers:

```powershell
# Navigate to database repository
cd path\to\final-project-database

# Extract OSM file (if compressed)
tar -xzf labeled.osm.tar.gz

# Copy OSM file to container
docker cp labeled.osm transport-db:/labeled.osm

# Install osm2pgrouting tool
docker exec transport-db bash -c "apt-get update && apt-get install -y osm2pgrouting"

# Import street network to pgRouting tables
docker exec transport-db bash -c "osm2pgrouting --f /labeled.osm --dbname transport_db --username postgres --password postgres --clean"
```

Expected output:
- **68,020** street segments imported (`ways` table)
- **48,095** vertices imported (`ways_vertices_pgr` table)
- Execution time: ~9 seconds

### 7. Verify pgRouting Tables

```powershell
# Check pgRouting tables
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways; SELECT COUNT(*) FROM ways_vertices_pgr;"

# Test routing between two vertices
docker exec transport-db psql -U postgres -d transport_db -c "SELECT SUM(edge.cost) AS distance_meters FROM pgr_dijkstra('SELECT gid AS id, source, target, cost, reverse_cost FROM ways', 1, 100, directed:=false) AS route JOIN ways AS edge ON route.edge = edge.gid;"
```

Expected:
- `ways`: 68,020 rows
- `ways_vertices_pgr`: 48,095 rows
- Routing query returns distance in meters

## pgAdmin Access

### Web Interface

1. Open browser: http://localhost:8080
2. Login credentials:
   - **Email**: admin@example.com
   - **Password**: admin

### Add Server Connection

1. In pgAdmin, right-click "Servers" → "Register" → "Server"
2. **General** tab:
   - Name: `transport-db`
3. **Connection** tab:
   - Host: `transport-db`
   - Port: `5432`
   - Maintenance database: `transport_db`
   - Username: `postgres`
   - Password: `postgres`
4. Click **Save**

### Sample Queries in pgAdmin

```sql
-- Find nearest stops to a location
SELECT stop_id, name, 
       ST_Distance(geom_4326, ST_SetSRID(ST_Point(29.95, 31.2), 4326)) AS distance_m
FROM stop
ORDER BY geom_4326 <-> ST_SetSRID(ST_Point(29.95, 31.2), 4326)
LIMIT 5;

-- Routes passing through a specific stop
SELECT r.name, r.mode, rs.stop_sequence
FROM route_stop rs
JOIN route r ON rs.route_id = r.route_id
WHERE rs.stop_id = (SELECT stop_id FROM stop WHERE name ILIKE '%محطة مصر%' LIMIT 1)
ORDER BY r.name;

-- Stops along a route
SELECT s.stop_id, s.name, rs.stop_sequence
FROM route_stop rs
JOIN stop s ON rs.stop_id = s.stop_id
WHERE rs.route_id = 1
ORDER BY rs.stop_sequence;
```

## Connect from Python Agent

### Environment Variables

Set these in PowerShell before running your agent:

```powershell
$env:DB_HOST = "localhost"
$env:DB_NAME = "transport_db"
$env:DB_USER = "postgres"
$env:DB_PASSWORD = "postgres"
$env:USE_DB = "1"
$env:DEBUG_ROUTING = "1"
```

### Activate Virtual Environment & Install Dependencies

**IMPORTANT**: Always use the virtual environment Python to avoid import errors.

```powershell
# Navigate to agent directory
cd C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\agent-mvp

# Activate virtual environment (use parent .venv)
cd ..
.\.venv\Scripts\Activate.ps1
cd agent-mvp

# Install required packages (only first time)
pip install psycopg2-binary langchain langchain-google-genai osmnx networkx geopy python-dotenv pandas
```

### Run Agent with DB-First Mode

**Option 1: Using activated venv**
```powershell
# After activating venv (see above)
python test_agent.py
```

**Option 2: Using full path (recommended)**
```powershell
& "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\.venv\Scripts\python.exe" test_agent.py
```

The agent will:
- **Parse intent** with LLM (Gemini) to extract origin/destination
- **Search stops** by name in database first (fast, accurate)
- **Find nearest stops** using PostGIS KNN (<10ms per query)
- **Compute journeys** from route/route_stop tables (0-transfer, 1-transfer)
- **Calculate walking distances** using pgRouting (optional, for transfers)
- **Format output** with final LLM call for natural Arabic response
- **Fall back** to OSM/Python pipeline if DB unavailable

## Maintenance Commands

### Restart Containers

```powershell
docker compose restart
```

### Stop and Remove Everything

```powershell
docker compose down -v
```

Warning: `-v` removes volumes (deletes all data). Use `docker compose down` to keep data.

### Rebuild After Changes

```powershell
docker compose down -v
docker compose up -d --build
# Then repeat steps 2-4 to reinitialize
```

### View Logs

```powershell
docker logs transport-db
docker logs transport-pgadmin
```

### Connect via psql from Host

If you have `psql` installed locally:

```powershell
psql -h localhost -U postgres -d transport_db
```

## Database Schema Overview

### Operational Tables

- **stop**: Stop locations with PostGIS geometries (`geom_4326`, `geom_22992`)
- **route**: Transport routes (buses, microbuses, etc.)
- **route_geometry**: Route line geometries (WGS84 + projected)
- **route_stop**: Stops along each route with sequences

### GTFS Staging Tables

- **gtfs_staging_agency**: Transit agencies
- **gtfs_staging_routes**: Route definitions
- **gtfs_staging_stops**: Stop locations (raw GTFS)
- **gtfs_staging_trips**: Individual trips
- **gtfs_staging_stop_times**: Stop times per trip
- **gtfs_staging_shapes**: Route geometries (raw)
- **gtfs_staging_calendar**: Service calendars
- **gtfs_staging_feed_info**: Feed metadata

### Extensions Installed

- **PostGIS 3.6**: Spatial database capabilities
- **pgRouting**: Routing and network analysis
- **pg_trgm**: Fuzzy text search
- **btree_gin**: GIN indexes on scalar types

## Troubleshooting

### Container Keeps Restarting

Check logs:
```powershell
docker logs transport-db
```

Common causes:
- Volume mount issues (Postgres 18 requires `/var/lib/postgresql`, not `/var/lib/postgresql/data`)
- CRLF line endings in scripts (run dos2unix fix)

### GTFS Import Fails

Error: `relation "gtfs_staging_agency" does not exist`
- Solution: Run schema loading scripts (step 3)

Error: `$'\r': command not found` or `invalid option name`
- Solution: Run dos2unix fix (step 2)

### Cannot Connect from Python

Error: `Connection refused`
- Check container is running: `docker ps`
- Check port 5432 is exposed: `docker ps -a` (should show `0.0.0.0:5432->5432/tcp`)

Error: `authentication failed`
- Verify credentials match docker-compose.yml defaults

### Tables Exist But Empty

Run the GTFS import:
```powershell
docker exec transport-db bash -c "gtfs2db.sh --with-etl"
```

## Database Connection Details

- **Host**: localhost (from host machine) or `transport-db` (from containers)
- **Port**: 5432
- **Database**: transport_db
- **Username**: postgres
- **Password**: postgres

## Data Flow

```
GTFS CSV Files (gtfs-data/)
     ↓
[gtfs2db.sh] → GTFS Staging Tables (raw data preserved)
     ↓
[ETL Transform] → Operational Schema (normalized, with PostGIS)
     ↓
OSM File (labeled.osm)
     ↓
[osm2pgrouting] → pgRouting Tables (ways, ways_vertices_pgr)
     ↓
Application Queries / Agent
```

## Agent Token Optimization 🚀

**How to minimize LLM token usage while using database and pgRouting:**

### Current Architecture (Efficient!)

The agent uses **only 2 LLM calls per query**:

1. **Parse intent** (small): Extract origin/destination from user query
   - Input: ~50 tokens (user query)
   - Output: ~20 tokens (JSON with locations)
   - Model: gemini-pro (fast, cheap)

2. **Final polish** (medium): Format journeys in natural Arabic
   - Input: ~200-500 tokens (journey data)
   - Output: ~300-800 tokens (formatted response)
   - Model: gemini-2.5-flash → gemini-pro fallback

**All routing logic uses database queries** (no tokens!):
- Stop name search: SQL ILIKE
- Nearest stop: PostGIS KNN
- Journey computation: SQL joins on route_stop
- Walking distance: pgr_dijkstra

### Total Token Usage Per Query

- **Best case**: ~600 tokens (parse + polish)
- **Typical**: ~1,000 tokens
- **Maximum**: ~1,500 tokens

Compare to old approach (10,000+ tokens with multiple LLM calls for routing logic).

### Tips to Further Reduce Tokens

1. **Cache common queries**: Store frequent routes in Redis/memory
2. **Batch queries**: Process multiple user queries in one session
3. **Skip final polish** (optional): Return raw journey JSON for APIs
4. **Use smaller models**: Switch gemini-2.5-flash → gemini-1.5-flash if available

### Environment Variables for Control

```powershell
# Essential
$env:USE_DB = "1"              # Use database (no fallback to OSM)
$env:DEBUG_ROUTING = "0"       # Disable debug prints (reduces output)

# Optional (advanced)
$env:SKIP_LLM_POLISH = "1"     # Skip final LLM formatting (future enhancement)
$env:CACHE_JOURNEYS = "1"      # Cache computed journeys (future enhancement)
```

## Complete Setup from Scratch

**Time required**: ~15-20 minutes

```powershell
# 1. Clone database repository
git clone https://github.com/marwan051/final-project-database
cd final-project-database

# 2. Start Docker containers
docker compose up -d --build

# 3. Fix line endings (Windows)
docker exec transport-db bash -c "apt-get update && apt-get install -y dos2unix"
docker exec transport-db bash -c "dos2unix /docker-entrypoint-initdb.d/*.sh /usr/local/bin/*.sh"

# 4. Initialize database
docker exec transport-db bash -c "/docker-entrypoint-initdb.d/01-init-database.sh"
docker exec transport-db bash -c "psql -U postgres -d transport_db -f /docker-entrypoint-initdb.d/02-schema.sql"
docker exec transport-db bash -c "psql -U postgres -d transport_db -f /docker-entrypoint-initdb.d/03-gtfs-staging-schema.sql"
docker exec transport-db bash -c "psql -U postgres -d transport_db -f /docker-entrypoint-initdb.d/04-gtfs-etl-transform.sql"

# 5. Import GTFS data
docker exec transport-db bash -c "gtfs2db.sh --with-etl"

# 6. Import OSM street network
tar -xzf labeled.osm.tar.gz
docker cp labeled.osm transport-db:/labeled.osm
docker exec transport-db bash -c "apt-get update && apt-get install -y osm2pgrouting"
docker exec transport-db bash -c "osm2pgrouting --f /labeled.osm --dbname transport_db --username postgres --password postgres --clean"

# 7. Verify everything
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM stop;"
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways;"

# 8. Set up Python agent
cd ..\agent-mvp
$env:DB_HOST = "localhost"
$env:DB_NAME = "transport_db"
$env:DB_USER = "postgres"
$env:DB_PASSWORD = "postgres"
$env:USE_DB = "1"
$env:DEBUG_ROUTING = "1"

# 9. Run agent (use full path to avoid import errors)
& "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\.venv\Scripts\python.exe" test_agent.py
```

**Expected results**:
- ✅ 441 stops, 104 routes imported
- ✅ 68,020 street segments imported
- ✅ Agent runs with 2 LLM calls per query
- ✅ Average response time: ~5-10 seconds

## Backup & Restore

### Backup Database

```powershell
docker exec transport-db pg_dump -U postgres -d transport_db -F c -f /tmp/backup.dump
docker cp transport-db:/tmp/backup.dump ./backup_$(Get-Date -Format 'yyyyMMdd').dump
```

### Restore Database

```powershell
docker cp backup_20251213.dump transport-db:/tmp/backup.dump
docker exec transport-db pg_restore -U postgres -d transport_db -c /tmp/backup.dump
```



