import os
import sys
import grpc
from typing import List, Dict, Any

# We expect generated stubs to live under `grpc_stubs/`
_STUBS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "grpc_stubs")
if _STUBS_DIR not in sys.path:
    sys.path.insert(0, _STUBS_DIR)
try:
    from grpc_stubs import routing_pb2, routing_pb2_grpc
except ImportError as e:
    raise ImportError(
        "gRPC stubs not found. Generate them via: "
        "python -m grpc_tools.protoc -I grpc --python_out grpc_stubs --grpc_python_out grpc_stubs grpc/routing.proto"
    ) from e

ROUTING_SERVER_ADDR = os.getenv("ROUTING_SERVER_ADDR", "localhost:50051")


def _get_stub():
    channel = grpc.insecure_channel(ROUTING_SERVER_ADDR)
    return routing_pb2_grpc.RoutingServiceStub(channel)


def health_check() -> Dict[str, Any]:
    stub = _get_stub()
    try:
        resp = stub.HealthCheck(routing_pb2.HealthRequest())
        return {"status": resp.status, "message": resp.message}
    except Exception as e:
        return {"status": "unreachable", "message": str(e)}


def find_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    walking_cutoff: float = 1000.0,
    max_transfers: int = 2,
) -> Dict[str, Any]:
    """Call the remote gRPC FindRoute and return parsed dict."""
    stub = _get_stub()
    req = routing_pb2.RouteRequest(
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        max_transfers=max_transfers,
        walking_cutoff=walking_cutoff,
    )
    try:
        resp = stub.FindRoute(req)
    except Exception as e:
        return {"num_journeys": 0, "journeys": [], "error": str(e)}

    journeys: List[Dict[str, Any]] = []
    for j in resp.journeys:
        journeys.append(
            {
                "path": list(j.path),
                "costs": {
                    "money": j.costs.money,
                    "transport_time": j.costs.transport_time,
                    "walk": j.costs.walk,
                },
            }
        )

    return {
        "num_journeys": resp.num_journeys,
        "journeys": journeys,
        "start_trips_found": resp.start_trips_found,
        "end_trips_found": resp.end_trips_found,
    }
