# gRPC Integration for agent-mvp

This agent connects to the Transit Routing gRPC server (`final_project_routing_server`) to find journeys.

## Setup

- Ensure Docker Desktop is running.
- Clone the server repo and start the container:

```powershell
# Clone server
git clone https://github.com/Marwan051/final_project_routing_server.git
cd final_project_routing_server

# Build and run (exposes gRPC on 50051)
docker build -t routing-server .
docker run --rm -p 50051:50051 routing-server
```

## Generate gRPC stubs (first time)

```powershell
# From the agent workspace root
Push-Location "c:/Users/Rowan/OneDrive/Desktop/collage/graduation project/geniAi/agent/agent-mvp"
python -m pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I grpc --python_out grpc_stubs --grpc_python_out grpc_stubs grpc/routing.proto
Pop-Location
```

## Configure

- Optionally set server address via environment:

```powershell
$env:ROUTING_SERVER_ADDR = "localhost:50051"
```

## Try the agent (single run)

```powershell
python test_agent.py
```

The agent will geocode, call `FindRoute`, and print a friendly Arabic explanation of the journey.
