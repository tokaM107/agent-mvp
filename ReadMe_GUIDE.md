
###  **Database Tables Ready**
The database will automatically create these tables on first run:
-  `route` - Transport routes (microbuses, buses, etc.)
-  `trip` - Specific scheduled trips
-  `stop` - Bus/microbus stops with GPS coordinates
-  `route_stop` - Stops in each trip with sequences
-  `route_geometry` - Route shapes/paths
-  Extensions: PostGIS, pgRouting, pg_trgm (for fuzzy search)

**pgRouting tables (need manual seeding):**
- ⚠️ `ways` - Road network segments (created by osm2pgrouting)
- ⚠️ `ways_vertices_pgr` - Road intersections (created by osm2pgrouting)

### 3. **Walking Distance Calculation**
- ✅ Code checks if `ways` tables exist before querying
- ✅ If tables exist: Real walking distances via pgRouting
- ⚠️ If tables DON'T exist: Walking distance = 0 (but agent still works)

### 4. **API Key**
- ✅ Stored in `.env`: `GOOGLE_API_KEY=AIzaSyD_KCJVRE61KorQDkqvscJ-x6VuCUZYVvo`
- ✅ docker-compose.yml reads from .env and passes to agent container
- ✅ Agent will see the API key automatically

### 5. **Pricing Model**
- ✅ Model file exists: `models/trip_price_model.joblib`
- ✅ `services/pricing.py` loads model on startup
- ✅ Calculates fare based on distance traveled

---

##  Step-by-Step Deployment

### **Step 1: Build & Start Containers**

**Navigate to the database folder:**
```powershell
cd "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\agent-mvp\final-project-database"
```

**Load API key from .env into PowerShell environment:**
```powershell
$env:GOOGLE_API_KEY = (Get-Content ..\.env | Select-String 'GOOGLE_API_KEY' | ForEach-Object { $_ -replace 'GOOGLE_API_KEY=', '' }).ToString().Trim()
```

**Build and start all services:**
```powershell
docker compose up -d --build
```

This will:
- Build the PostgreSQL database image (with osm2pgrouting now!)
- Build the Python agent image
- Start 3 containers: `transport-db`, `transport-pgadmin`, `transport-agent`
- Load GTFS data automatically (routes, stops, trips)
- Pass API key from environment to agent container
- Create all database tables

**Verify containers are running:**
```powershell
docker ps
```
You should see:
- `transport-db` - PostgreSQL database
- `transport-pgadmin` - Web UI on http://localhost:8080
- `transport-agent` - Python agent

---

### **Step 2: Verify GTFS Data Loaded**

**Check if stops and routes exist:**
```powershell
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM stop;"
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM route;"
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM trip;"
```

Expected output:
- Stops: Should show number > 0 
- Routes: Should show number > 0
- Trips: Should show number > 0 

If counts are 0, GTFS data didn't load. Check:
```powershell
docker logs transport-db | Select-String "GTFS"
```

---

### **Step 3: Seed pgRouting Tables (REQUIRED for Walking Distances)**

**Make sure `labeled.osm` is in the agent-mvp folder:**
```powershell
cd "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\agent-mvp"
ls labeled.osm  # Should show the file
```

**Copy OSM file to database container:**
```powershell
docker cp labeled.osm transport-db:/tmp/labeled.osm
```

**Run osm2pgrouting to create routing tables:**
```powershell
docker exec -it transport-db osm2pgrouting `
  -f /tmp/labeled.osm `
  -d transport_db `
  -U postgres `
  -h localhost `
  -W postgres `
  --clean
```

**Enter password when prompted:** `postgres`

This creates the `ways` and `ways_vertices_pgr` tables needed for walking distance calculation.

**Verify routing tables exist:**
```powershell
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways;"
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways_vertices_pgr;"
```

Expected output: Both should show numbers > 0

---

### **Step 4: Run the Agent**

**From the database folder (same directory as docker-compose.yml):**
```powershell
docker exec transport-agent python test_agent.py
```

**Or run with specific query:**
```powershell
docker exec transport-agent python -c "import test_agent; print(test_agent.run_once('أريد الذهاب من العصافرة إلى المنشية'))"
```

**Verify API key is loaded (optional check):**
```powershell
docker exec transport-agent bash -c 'echo $GOOGLE_API_KEY'
```

Expected output: Your API key from .env file

---

## 🔄 Updating API Key

If you need to change the API key:

**1. Edit .env file:**
```powershell
notepad "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\agent-mvp\.env"
```

**2. Load new key into PowerShell environment:**
```powershell
cd "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\agent-mvp\final-project-database"
$env:GOOGLE_API_KEY = (Get-Content ..\.env | Select-String 'GOOGLE_API_KEY' | ForEach-Object { $_ -replace 'GOOGLE_API_KEY=', '' }).ToString().Trim()
```

**3. Restart containers to reload environment:**
```powershell
docker compose down
docker compose up -d
```

**4. Verify new key loaded:**
```powershell
docker exec transport-agent bash -c 'echo $GOOGLE_API_KEY'
```

---

### **Step 5 (Optional): Verify Everything**

**Check database has data:**
```powershell
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM stop;"
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM route;"
```

**Check pgRouting tables (if seeded):**
```powershell
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways;"
docker exec transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways_vertices_pgr;"
```

**Check container status:**
```powershell
docker ps
```

Should show 3 containers running:
- `transport-db` - Database
- `transport-agent` - Python agent
- `transport-pgadmin` - Web UI (http://localhost:8080)

---

## 📋 Complete Quick Reference

**From `final-project-database/` folder:**

```powershell
# 1. Navigate to database folder
cd "C:\Users\Rowan\OneDrive\Desktop\collage\graduation project\geniAi\agent\agent-mvp\final-project-database"

# 2. Load API key from .env
$env:GOOGLE_API_KEY = (Get-Content ..\.env | Select-String 'GOOGLE_API_KEY' | ForEach-Object { $_ -replace 'GOOGLE_API_KEY=', '' }).ToString().Trim()

# 3. Start containers
docker compose up -d --build

# 4. Seed pgRouting (first time only, or after data changes)
docker cp ..\labeled.osm transport-db:/tmp/labeled.osm
docker exec -it transport-db osm2pgrouting -f /tmp/labeled.osm -d transport_db -U postgres -h localhost -W postgres --clean
# Enter password: postgres

# 5. Run agent
docker exec transport-agent python test_agent.py
```

---


### **Without pgRouting Seeded (Skip Step 3):**
⚠️ Walking distance will be 0
⚠️ Agent still finds routes but less accurate
⚠️ Recommendation: ALWAYS run Step 3

---

##  Troubleshooting

### Problem: "Your API key was reported as leaked"

**Solution:**
1. Get new API key from: https://aistudio.google.com/app/apikey
2. Update `.env` file with new key
3. **IMPORTANT:** Load new key and restart containers:
   ```powershell
   cd final-project-database
   $env:GOOGLE_API_KEY = (Get-Content ..\.env | Select-String 'GOOGLE_API_KEY' | ForEach-Object { $_ -replace 'GOOGLE_API_KEY=', '' }).ToString().Trim()
   docker compose down
   docker compose up -d
   ```
4. Verify new key loaded:
   ```powershell
   docker exec transport-agent bash -c 'echo $GOOGLE_API_KEY'
   ```

**Note:** Simply restarting with `docker compose restart` won't reload the .env file - you MUST use `docker compose down` then `up`.

### Problem: "No containers running"
```powershell
docker compose up -d
```

### Problem: "Database connection failed"
```powershell
# Check if DB is healthy
docker exec transport-db pg_isready -U postgres -d transport_db

# Restart if needed
docker restart transport-db
```

### Problem: "GTFS tables empty"
```powershell
# Check logs for ETL process
docker logs transport-db | Select-String "GTFS|ETL|error"

# Manually trigger ETL if needed
docker exec -it transport-db bash /usr/local/bin/init-gtfs.sh
```

### Problem: "osm2pgrouting not found"
This means Docker wasn't rebuilt with new Dockerfile. Run:
```powershell
cd final-project-database
docker compose down
docker compose up -d --build
```

### Problem: "API key not visible in agent"
```powershell
# Check .env exists
cat ..\.env | Select-String "GOOGLE_API_KEY"

# Restart agent with explicit key
docker exec -e GOOGLE_API_KEY=$Env:GOOGLE_API_KEY transport-agent python test_agent.py
```

### Problem: "ways table does not exist"
You skipped Step 3. Run osm2pgrouting seeding.

---

##  Folder Structure Guide

```
agent-mvp/                           ← Main agent code
├── .env                            ← API key stored here
├── labeled.osm                     ← OSM data for routing
├── models/
│   └── trip_price_model.joblib    ← Pricing model ( exists)
├── services/
│   ├── pricing.py                 ← Fare calculation
│   └── distance.py                ← Distance calculation
├── tools.py                        ← Agent tools (fixed SQL here)
├── test_agent.py                  ← Run this to test
└── final-project-database/         ← Database setup folder
    ├── docker-compose.yml          ← API key passed here
    ├── Dockerfile                  ← osm2pgrouting added here
    ├── gtfs-data/                  ← GTFS CSV files
    └── sql/
        └── schema.sql              ← Table definitions
```

**Where to run commands:**
- **Docker commands:** Run from `final-project-database/` folder
- **Agent commands:** Run from `agent-mvp/` folder (or use docker exec)

---

##  Quick Checklist

Before running agent, verify:

- [ ] Docker Desktop running
- [ ] `.env` file exists in agent-mvp/ with API key
- [ ] `labeled.osm` exists in agent-mvp/
- [ ] Containers built: `docker ps` shows 3 containers
- [ ] GTFS data loaded: `SELECT COUNT(*) FROM stop;` > 0
- [ ] pgRouting seeded: `SELECT COUNT(*) FROM ways;` > 0
- [ ] API key visible: `docker exec transport-agent env | Select-String GOOGLE_API_KEY`

---


**Do you need to rebuild?**
**YES** - Because we added osm2pgrouting to Dockerfile

**Run from `final-project-database/` folder:**
```powershell
docker compose down
docker compose up -d --build
```

