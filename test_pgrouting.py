"""Test pgRouting integration for walking distance computation."""

import psycopg2
import os


def get_db_connection():
    """Create database connection."""
    host = os.environ.get("DB_HOST", "localhost")
    db = os.environ.get("DB_NAME", "transport_db")
    user = os.environ.get("DB_USER", "postgres")
    pwd = os.environ.get("DB_PASSWORD", "postgres")
    try:
        return psycopg2.connect(host=host, database=db, user=user, password=pwd)
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def snap_stop_to_vertex_db(stop_id: int):
    """Find nearest pgRouting vertex to a stop."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        q = (
            "SELECT v.id FROM ways_vertices_pgr v, stop s "
            "WHERE s.stop_id = %s "
            "ORDER BY v.the_geom <-> s.geom_4326 LIMIT 1;"
        )
        cur.execute(q, (stop_id,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        print(f"Snap error: {e}")
        return None
    finally:
        conn.close()


def compute_walk_distance_db(stop_id_a: int, stop_id_b: int):
    """Compute walking distance using pgr_dijkstra."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        vertex_a = snap_stop_to_vertex_db(stop_id_a)
        vertex_b = snap_stop_to_vertex_db(stop_id_b)
        if not vertex_a or not vertex_b:
            return None
        
        cur = conn.cursor()
        q = (
            "SELECT SUM(edge.cost) AS total_distance "
            "FROM pgr_dijkstra("
            "  'SELECT gid AS id, source, target, cost, reverse_cost FROM ways', "
            "  %s, %s, directed:=false"
            ") AS route "
            "JOIN ways AS edge ON route.edge = edge.gid;"
        )
        cur.execute(q, (vertex_a, vertex_b))
        row = cur.fetchone()
        return float(row[0]) if row and row[0] else None
    except Exception as e:
        print(f"Walk distance error: {e}")
        return None
    finally:
        conn.close()


def test_database_connection():
    """Test database connection."""
    print("=" * 60)
    print("Test 1: Database Connection")
    print("=" * 60)
    
    conn = get_db_connection()
    if conn:
        print("✅ Database connection successful")
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM stop;")
            stop_count = cur.fetchone()[0]
            print(f"✅ Found {stop_count} stops in database")
            conn.close()
        except Exception as e:
            print(f"❌ Error querying database: {e}")
    else:
        print("❌ Database connection failed")
    print()


def test_pgrouting_tables():
    """Test pgRouting tables existence."""
    print("=" * 60)
    print("Test 2: pgRouting Tables")
    print("=" * 60)
    
    conn = get_db_connection()
    if not conn:
        print("❌ No database connection")
        return
    
    try:
        cur = conn.cursor()
        
        # Check ways table
        cur.execute("SELECT COUNT(*) FROM ways;")
        ways_count = cur.fetchone()[0]
        print(f"✅ Ways table: {ways_count:,} street segments")
        
        # Check vertices table
        cur.execute("SELECT COUNT(*) FROM ways_vertices_pgr;")
        vertices_count = cur.fetchone()[0]
        print(f"✅ Vertices table: {vertices_count:,} vertices")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error checking pgRouting tables: {e}")
    print()


def test_snap_stop_to_vertex():
    """Test snapping stops to nearest street vertices."""
    print("=" * 60)
    print("Test 3: Snap Stop to Vertex")
    print("=" * 60)
    
    # Test with first few stops
    test_stop_ids = [1, 2, 3]
    
    for stop_id in test_stop_ids:
        vertex_id = snap_stop_to_vertex_db(stop_id)
        if vertex_id:
            print(f"✅ Stop {stop_id} → Vertex {vertex_id}")
        else:
            print(f"❌ Stop {stop_id} → No vertex found")
    print()


def test_walking_distance():
    """Test walking distance computation between stops."""
    print("=" * 60)
    print("Test 4: Walking Distance Computation")
    print("=" * 60)
    
    # Get stop names first
    conn = get_db_connection()
    if not conn:
        print("❌ No database connection")
        return
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT stop_id, name FROM stop LIMIT 5;")
        stops = cur.fetchall()
        conn.close()
        
        if len(stops) < 2:
            print("❌ Not enough stops to test")
            return
        
        # Test walking distance between first two stops
        stop1, name1 = stops[0]
        stop2, name2 = stops[1]
        
        print(f"\nComputing walk distance between:")
        print(f"  Stop {stop1}: {name1}")
        print(f"  Stop {stop2}: {name2}")
        print()
        
        walk_meters = compute_walk_distance_db(stop1, stop2)
        
        if walk_meters:
            walk_km = walk_meters / 1000
            print(f"✅ Walking distance: {walk_meters:.0f} meters ({walk_km:.2f} km)")
            
            # Estimate walking time (avg 5 km/h = 83.33 m/min)
            walk_minutes = walk_meters / 83.33
            print(f"   Estimated walking time: {walk_minutes:.1f} minutes")
        else:
            print("❌ No walking path found")
            print("   (Possible reasons: stops too far apart, disconnected street network)")
    
    except Exception as e:
        print(f"❌ Error testing walking distance: {e}")
    print()


def test_nearby_stops_walk():
    """Test walking distance between nearby stops."""
    print("=" * 60)
    print("Test 5: Walking Distance for Nearby Stops")
    print("=" * 60)
    
    conn = get_db_connection()
    if not conn:
        print("❌ No database connection")
        return
    
    try:
        cur = conn.cursor()
        # Find two stops that are close to each other (within 1km)
        q = """
        SELECT 
            s1.stop_id AS stop1_id,
            s1.name AS stop1_name,
            s2.stop_id AS stop2_id,
            s2.name AS stop2_name,
            ST_Distance(
                ST_Transform(s1.geom_4326, 3857),
                ST_Transform(s2.geom_4326, 3857)
            ) AS straight_line_meters
        FROM stop s1, stop s2
        WHERE s1.stop_id < s2.stop_id
        AND ST_DWithin(
            ST_Transform(s1.geom_4326, 3857),
            ST_Transform(s2.geom_4326, 3857),
            1000  -- 1km radius
        )
        ORDER BY straight_line_meters
        LIMIT 1;
        """
        cur.execute(q)
        result = cur.fetchone()
        conn.close()
        
        if not result:
            print("❌ No nearby stops found within 1km")
            return
        
        stop1_id, stop1_name, stop2_id, stop2_name, straight_line = result
        
        print(f"\nTesting nearby stops:")
        print(f"  Stop {stop1_id}: {stop1_name}")
        print(f"  Stop {stop2_id}: {stop2_name}")
        print(f"  Straight-line distance: {straight_line:.0f} meters")
        print()
        
        walk_meters = compute_walk_distance_db(stop1_id, stop2_id)
        
        if walk_meters:
            walk_km = walk_meters / 1000
            detour_ratio = walk_meters / straight_line
            print(f"✅ Walking distance: {walk_meters:.0f} meters ({walk_km:.2f} km)")
            print(f"   Detour factor: {detour_ratio:.2f}x (1.0 = straight line)")
            
            walk_minutes = walk_meters / 83.33
            print(f"   Estimated walking time: {walk_minutes:.1f} minutes")
        else:
            print("❌ No walking path found")
    
    except Exception as e:
        print(f"❌ Error testing nearby stops: {e}")
    print()


def main():
    """Run all pgRouting tests."""
    print("\n" + "=" * 60)
    print("pgRouting Integration Test Suite")
    print("=" * 60)
    print()
    
    test_database_connection()
    test_pgrouting_tables()
    test_snap_stop_to_vertex()
    test_walking_distance()
    test_nearby_stops_walk()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
