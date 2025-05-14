"""
Beispiel zum Starten des Ethics Model API-Servers.

Dieses Skript demonstriert, wie man den Ethics Model API-Server startet
und verschiedene Konfigurationsoptionen verwendet.
"""

import os
import argparse
import uvicorn

# Umgebungsvariablen für API-Konfiguration setzen
os.environ["ETHICS_API_PORT"] = "8000"
os.environ["ETHICS_API_DEBUG"] = "True"
os.environ["ETHICS_API_MAX_SEQUENCE_LENGTH"] = "256"

# Optional: Pfad zum Modell-Checkpoint setzen 
# os.environ["ETHICS_API_CHECKPOINT_PATH"] = "checkpoints/model.pt"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Ethics Model API server")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host address to bind"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    
    return parser.parse_args()

def main():
    """Run the API server."""
    args = parse_args()
    
    print(f"Starte Ethics Model API-Server auf http://{args.host}:{args.port}")
    print("Drücken Sie STRG+C zum Beenden.")
    
    # API starten
    uvicorn.run(
        "ethics_model.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
