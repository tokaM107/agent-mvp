"""
Direct test of gRPC routing without the agent framework.
This bypasses LangChain to verify the server and tools work.
"""
from services.routing_client import find_route, health_check
from services.geocode import geocode_address as geocode_fn

print("="*60)
print("Testing Direct gRPC Call (No Agent)")
print("="*60)

# Test 0: Health check
print("\n0. Health check...")
health = health_check()
print(f"  Status: {health.get('status')}")
print(f"  Message: {health.get('message')}")

# Test 1: Direct gRPC call with known coordinates
print("\n1. Testing with known-good coordinates...")
# Coordinates from server README (Raml station area to Sidi Gabir area)
try:
    result = find_route(
        start_lat=31.22968895248673,
        start_lon=29.96139328537071,
        end_lat=31.20775934404925,
        end_lon=29.94194179397711,
        walking_cutoff=1500.0,
        max_transfers=2
    )
    
    print(f"  Journeys found: {result.get('num_journeys', 0)}")
    print(f"  Start trips: {result.get('start_trips_found', 0)}")
    print(f"  End trips: {result.get('end_trips_found', 0)}")
    
    if result.get('num_journeys', 0) > 0:
        first_journey = result['journeys'][0]
        print(f"\n  First journey preview:")
        print(f"    Path: {' → '.join(first_journey['path'][:3])}...")
        print(f"    Cost: {first_journey['costs']['money']:.2f} EGP")
        print(f"    Walk: {int(first_journey['costs']['walk'])} m")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*60)
print("If you see journeys above, the gRPC server is working!")
print("The agent timeout issue is likely with Gemini API.")
print("="*60)
