# Setup Instructions

## 1. Generate gRPC Stubs
```bash
python -m grpc_tools.protoc -I. --python_out=./grpc_stubs --grpc_python_out=./grpc_stubs grpc/routing.proto
```

## 2. Build Routing Server (from final_project_routing_server)
```bash
cd final_project_routing_server
docker build -t routing-server .
cd ..
```

## 3. Run Routing Server
```bash
docker run --rm -p 50051:50051 routing-server
```

## 4. Install Python Dependencies (in another terminal)
```bash
pip install -r requirements.txt
```

## 5. Create .env File
```bash
echo GEMINI_API_KEY=your_api_key_here > .env
```

## 6. Run Agent
```bash
python test_agent.py
```

## Health Check
```bash
python -c "from services.routing_client import health_check; print(health_check())"
```

## Stop Container
```bash
docker stop routing-server
```
