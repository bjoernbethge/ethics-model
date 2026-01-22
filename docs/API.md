# API Documentation

The Ethics Model API is a FastAPI-based service for analyzing texts for ethical content, manipulation techniques, and moral frameworks.

## Features

- **Real-time Text Analysis**: Analyze texts for ethical aspects and manipulation techniques
- **Batch Processing**: Process multiple texts simultaneously for efficient analysis
- **Asynchronous Processing**: Long analyses can be performed asynchronously
- **Comprehensive Results**: Detailed analysis of moral frameworks, narrative framing, and manipulation techniques
- **Training Endpoints**: Train and fine-tune models via REST API
- **Visualization**: Generate interactive visualizations of analysis results
- **Automatic Documentation**: OpenAPI/Swagger documentation for easy integration

## Installation

No additional installation necessary. The API is part of the `ethics_model` package.

## Quick Start

### Starting the API Server

```bash
# Method 1: Using Python module
python -m ethics_model.api.run

# Method 2: Using the run script from examples
python examples/run_api_server.py

# Method 3: Direct from Python code
from ethics_model.api import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

## Configuration

Configure the API using environment variables or command-line arguments:

```bash
# Environment variables
export ETHICS_API_PORT=8080
export ETHICS_API_DEBUG=True
export ETHICS_API_CHECKPOINT_PATH=/path/to/checkpoint.pt
export ETHICS_API_HOST=0.0.0.0

# Or via command line
python -m ethics_model.api.run --port 8080 --debug --checkpoint-path /path/to/checkpoint.pt
```

### Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| ETHICS_API_HOST | Host address | 0.0.0.0 |
| ETHICS_API_PORT | Port number | 8000 |
| ETHICS_API_DEBUG | Enable debug mode | False |
| ETHICS_API_MODEL_PATH | Path to model directory | None |
| ETHICS_API_CHECKPOINT_PATH | Path to model checkpoint | None |
| ETHICS_API_TOKENIZER_NAME | Tokenizer to use | gpt2 |
| ETHICS_API_LLM_NAME | LLM model to use | gpt2 |
| ETHICS_API_DEVICE | Device for model execution (cuda/cpu) | Auto |
| ETHICS_API_MAX_SEQUENCE_LENGTH | Maximum sequence length | 512 |

## API Endpoints

### Core Analysis Endpoints

#### `POST /analyze`
Analyze a single text for ethical content and manipulation.

**Request Body:**
```json
{
  "text": "Companies should prioritize profit above all else.",
  "include_details": true
}
```

**Response:**
```json
{
  "ethics_score": 0.35,
  "manipulation_score": 0.67,
  "dominant_framework": "utilitarianism",
  "frameworks": {
    "deontology": 0.15,
    "virtue_ethics": 0.25,
    "utilitarianism": 0.60
  },
  "manipulation_techniques": ["appeal_to_profit", "oversimplification"],
  "confidence": 0.82
}
```

#### `POST /analyze/batch`
Analyze multiple texts in a single request.

**Request Body:**
```json
{
  "texts": [
    "Helping others is always the right thing to do.",
    "The end justifies the means."
  ],
  "include_details": false
}
```

**Response:**
```json
{
  "results": [
    {
      "ethics_score": 0.85,
      "manipulation_score": 0.15,
      "dominant_framework": "virtue_ethics"
    },
    {
      "ethics_score": 0.42,
      "manipulation_score": 0.58,
      "dominant_framework": "consequentialism"
    }
  ]
}
```

#### `POST /analyze/async`
Start an asynchronous text analysis.

**Request Body:**
```json
{
  "text": "Long text to analyze...",
  "include_details": true
}
```

**Response:**
```json
{
  "task_id": "abc123",
  "status": "processing"
}
```

#### `GET /tasks/{task_id}`
Check the status of an asynchronous task.

**Response:**
```json
{
  "task_id": "abc123",
  "status": "completed",
  "result": {
    "ethics_score": 0.75,
    "manipulation_score": 0.25
  }
}
```

### Information Endpoints

#### `GET /`
Get API status and information.

**Response:**
```json
{
  "name": "Ethics Model API",
  "version": "0.1.0",
  "status": "running"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true
}
```

#### `GET /frameworks`
Get information about supported moral frameworks.

**Response:**
```json
{
  "frameworks": [
    {
      "name": "deontology",
      "description": "Rule-based ethical framework",
      "examples": ["duty", "rights", "rules"]
    },
    {
      "name": "virtue_ethics",
      "description": "Character-based ethical framework",
      "examples": ["courage", "honesty", "compassion"]
    }
  ]
}
```

#### `GET /manipulation-techniques`
Get information about detectable manipulation techniques.

**Response:**
```json
{
  "techniques": [
    {
      "name": "appeal_to_emotion",
      "description": "Using emotional appeals to persuade",
      "examples": ["fear-mongering", "guilt-tripping"]
    }
  ]
}
```

### Training Endpoints

#### `POST /training/train`
Start an asynchronous model training session.

**Request Body:**
```json
{
  "train_texts": ["text1", "text2"],
  "ethics_labels": [0.8, 0.2],
  "manipulation_labels": [0.1, 0.9],
  "epochs": 5,
  "batch_size": 8,
  "learning_rate": 0.0001
}
```

**Response:**
```json
{
  "task_id": "train_xyz789",
  "status": "started"
}
```

#### `GET /training/train/{task_id}`
Check the status of a training task.

**Response:**
```json
{
  "task_id": "train_xyz789",
  "status": "training",
  "progress": 0.45,
  "current_epoch": 3,
  "total_epochs": 5,
  "metrics": {
    "loss": 0.345,
    "accuracy": 0.78
  }
}
```

#### `GET /training/train/{task_id}/result`
Get the results of a completed training task.

**Response:**
```json
{
  "task_id": "train_xyz789",
  "status": "completed",
  "final_metrics": {
    "loss": 0.234,
    "accuracy": 0.85
  },
  "checkpoint_path": "/path/to/checkpoint.pt"
}
```

### Visualization Endpoints

#### `POST /visualization/visualize`
Create visualizations of model analysis.

**Request Body:**
```json
{
  "text": "Companies should prioritize profit.",
  "visualization_type": "frameworks"
}
```

**Response:**
```json
{
  "visualization_type": "frameworks",
  "data": {
    "plotly_json": "..."
  }
}
```

## Python Client

Use the provided Python client for easy integration:

```python
from ethics_model.api.client import EthicsModelClient

# Initialize client
client = EthicsModelClient(base_url="http://localhost:8000")

# Check API availability
if client.ping():
    # Analyze text
    result = client.analyze(
        "Companies should prioritize profit above all else.",
        include_details=True
    )
    
    print(f"Ethics Score: {result['ethics_score']}")
    print(f"Manipulation Score: {result['manipulation_score']}")
    print(f"Dominant Framework: {result['dominant_framework']}")
    
    # Batch analysis
    results = client.analyze_batch([
        "Text 1",
        "Text 2"
    ])
    
    # Start async analysis
    task_id = client.analyze_async("Long text...")
    
    # Check task status
    status = client.get_task_status(task_id)
```

## cURL Examples

### Analyze Text

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Companies should prioritize profit above all else.",
    "include_details": true
  }'
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2"],
    "include_details": false
  }'
```

### Start Training

```bash
curl -X POST "http://localhost:8000/training/train" \
  -H "Content-Type: application/json" \
  -d '{
    "train_texts": ["text1", "text2"],
    "ethics_labels": [0.8, 0.2],
    "manipulation_labels": [0.1, 0.9],
    "epochs": 5
  }'
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding:
- API keys
- OAuth2
- JWT tokens
- Rate limiting

## Rate Limiting

No rate limiting is currently implemented. For production use, consider adding rate limiting middleware.

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (endpoint or resource not found)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

Error responses include:
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## Performance Optimization

For optimal performance:

1. **Use batch processing** for multiple texts
2. **Enable GPU acceleration** via CUDA
3. **Use async endpoints** for long-running analyses
4. **Cache results** when possible
5. **Use appropriate batch sizes** for training

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t ethics-model-api .

# Run container
docker run -p 8000:8000 ethics-model-api
```

### Production Deployment

For production deployment, consider:

1. Using a reverse proxy (nginx, traefik)
2. Adding SSL/TLS certificates
3. Implementing authentication
4. Adding monitoring and logging
5. Using multiple workers
6. Implementing health checks

Example with multiple workers:

```bash
uvicorn ethics_model.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

## Integration Examples

### Web Application Integration

```javascript
// JavaScript/TypeScript example
async function analyzeText(text) {
  const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      include_details: true
    })
  });
  
  const result = await response.json();
  return result;
}
```

### Microservices Integration

```python
import requests

class EthicsAnalysisService:
    def __init__(self, api_url):
        self.api_url = api_url
    
    def analyze(self, text):
        response = requests.post(
            f"{self.api_url}/analyze",
            json={"text": text, "include_details": True}
        )
        return response.json()
```

## Troubleshooting

### API Won't Start

- Check if port 8000 is available
- Verify all dependencies are installed
- Check CUDA availability if using GPU
- Review error logs

### Model Not Loading

- Verify checkpoint path is correct
- Ensure checkpoint file is compatible
- Check available memory
- Verify model configuration

### Slow Response Times

- Enable GPU acceleration
- Use batch processing
- Reduce sequence length
- Use async endpoints for long texts

## Next Steps

- See [Training Guide](./TRAINING.md) for training your own models
- See [Model Architecture](./ARCHITECTURE.md) for technical details
- Check examples in `examples/` directory

## Support

For issues and questions:
- GitHub Issues: https://github.com/bjoernbethge/ethics-model/issues
- Documentation: https://github.com/bjoernbethge/ethics-model
