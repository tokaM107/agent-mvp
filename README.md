
## Prerequisites

- A valid Gemini API key

## Folder Layout
- final-project-database/ → Dockerized PostgreSQL + PostGIS + pgRouting + GTFS ETL "I cloned and forked it applied some changes in it "
- agent-mvp/ → Agent code (another folder "this repo main") and Dockerfile setup

## 1) Configure environment
Create `.env` in `agent-mvp/`:

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
USE_DB=1
DB_HOST=localhost
DB_PORT=5432
DB_NAME=transport_db
DB_USER=postgres
DB_PASSWORD=postgres
```

## 2) Start the database + pgAdmin + agent
From `agent-mvp/final-project-database/`:

```powershell
# Build and start
docker compose up -d --build

# Verify containers
docker ps
```

Compose services:
- `transport-db`: Postgres 18 + PostGIS + pgRouting (GTFS ETL runs on first start)
- `transport-pgadmin`: pgAdmin on http://localhost:8080 (admin@example.com / admin)
- `transport-agent`: Python 3.12 container mounting `agent-mvp/` as `/app`

The agent is intentionally started idle (`tail -f /dev/null`) so you can run commands manually.

## 3) Seed pgRouting (REQUIRED for walking distances)
The agent calculates walking distances using pgRouting. You **must** import OSM data using `osm2pgrouting` or journeys will have `walk = 0`.

### Prerequisites:
- Place `labeled.osm` file in the `agent-mvp` folder (same level as `main.py`)
- The database must be running (`docker ps` should show `transport-db`)

### Seeding Steps:

```powershell
# 1. Copy labeled.osm into the DB container
docker cp labeled.osm transport-db:/tmp/labeled.osm

# 2. Run osm2pgrouting inside the DB container to create routing tables
docker exec -it transport-db osm2pgrouting `
  -f /tmp/labeled.osm `
  -d transport_db `
  -U postgres `
  -h localhost `
  -W postgres `
  --clean
```

This creates two tables:
- `ways`: Road network segments with geometry
- `ways_vertices_pgr`: Network vertices (intersections)

### Verify Seeding:
```powershell
# Check if routing tables exist and have data
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways;"
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT COUNT(*) FROM ways_vertices_pgr;"
```

**Important Notes:**
- Without seeding, `find_journeys_db` will detect missing tables and skip walk distance calculations
- The `--clean` flag removes existing routing tables before importing
- If seeding fails, check that `labeled.osm` contains valid OpenStreetMap data with road networks

## 4) Run the agent
Inside the agent container:

```powershell
# Pass API key explicitly to exec (ensures parsing works)
docker exec -e GOOGLE_API_KEY=$Env:GOOGLE_API_KEY transport-agent python test_agent.py
```

```
	agent-app:
		environment:
			- GOOGLE_API_KEY=${GOOGLE_API_KEY}
```

If you run manually without compose env propagation, always use `-e GOOGLE_API_KEY=...` with `docker exec`.

## 5) What changed and why
- Added `agent-app` service to `docker-compose.yml` so the agent and DB share the same Docker network. This avoids Windows host networking issues.

- `test_agent.py` now runs in DB-only mode: no OSM graph initialization, no graph-based fallback.

## 6) Useful commands
```powershell
# View recent agent logs

# Run agent with a specific query
docker exec -e GOOGLE_API_KEY=$Env:GOOGLE_API_KEY -it transport-agent python -c "\
import test_agent; print(test_agent.run_once('أريد الذهاب من الموقف الجديد الي العصافرة'))"

# Quick DB sanity check
docker exec -it transport-db psql -U postgres -d transport_db -c "SELECT count(*) FROM stop;"
```

## 7) Troubleshooting
- If Gemini parsing fails, verify `GOOGLE_API_KEY` is present in the container:

```powershell
docker exec transport-agent env | findstr GOOGLE_API_KEY
```


```powershell
$Env:GOOGLE_API_KEY = (Get-Content ..\.env | Select-String 'GOOGLE_API_KEY' | % { $_ -replace 'GOOGLE_API_KEY=', '' })
```

## 8) Why the agent runs next to the DB
Running the Python agent inside Docker alongside the DB solves host networking and authentication issues on Windows. Both containers communicate over the Docker bridge network by service name (`DB_HOST=transport-db`), ensuring reliable connectivity independent of the host OS configuration.