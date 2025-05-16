# Ethics Model API

Diese API bietet einen FastAPI-basierten Service für das Ethics Model zum Analysieren von Texten auf ethische Inhalte, Manipulationstechniken und moralische Frameworks.

## Funktionen

- **Echtzeit-Textanalyse**: Analysieren Sie Texte auf ethische Aspekte und Manipulationstechniken
- **Batch-Verarbeitung**: Verarbeiten Sie mehrere Texte gleichzeitig für effiziente Analyse
- **Asynchrone Verarbeitung**: Lange Analysen können asynchron durchgeführt werden
- **Umfassende Ergebnisse**: Detaillierte Analyse von moralischen Frameworks, Narrativ-Framing und Manipulationstechniken
- **Automatische Dokumentation**: OpenAPI-/Swagger-Dokumentation zur einfachen Integration

## Installation

Keine zusätzliche Installation notwendig. Die API ist Teil des `ethics_model`-Pakets.

## Verwendung

### API-Server starten

```bash
# Methode 1: Python-Modul
python -m ethics_model.api.run

# Methode 2: Direkt aus dem Python-Code
from ethics_model.api import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Anpassung der Konfiguration

Sie können die API über Umgebungsvariablen oder Kommandozeilenargumente konfigurieren:

```bash
# Umgebungsvariablen
export ETHICS_API_PORT=8080
export ETHICS_API_DEBUG=True
export ETHICS_API_CHECKPOINT_PATH=/path/to/checkpoint.pt

# Oder über die Kommandozeile
python -m ethics_model.api.run --port 8080 --debug --checkpoint-path /path/to/checkpoint.pt
```

### Verwendung des Python-Clients

```python
from ethics_model.api.client import EthicsModelClient

# Client initialisieren
client = EthicsModelClient(base_url="http://localhost:8000")

# API-Verfügbarkeit prüfen
if client.ping():
    # Text analysieren
    result = client.analyze(
        "Unternehmen sollten Profit über alles andere stellen, ungeachtet ethischer Bedenken.",
        include_details=True
    )
    
    # Ergebnisse ausgeben
    print(f"Ethik-Score: {result['ethics_score']}")
    print(f"Manipulations-Score: {result['manipulation_score']}")
    print(f"Dominantes Framework: {result['dominant_framework']}")
    
    # Visualisierungen erstellen
    viz_data = client.visualize(
        "Unternehmen sollten Profit über alles andere stellen, ungeachtet ethischer Bedenken.",
        visualization_type="frameworks"
    )
    
    # Modell trainieren
    train_texts = ["Text 1", "Text 2", "Text 3"]
    ethics_labels = [0.8, 0.2, 0.5]
    manipulation_labels = [0.1, 0.9, 0.4]
    
    task_id = client.train(
        train_texts=train_texts,
        ethics_labels=ethics_labels,
        manipulation_labels=manipulation_labels,
        epochs=5
    )
    
    # Trainingsstatus überprüfen
    status = client.get_training_status(task_id)
    print(f"Training Status: {status['status']}, Fortschritt: {status['progress']}")
```

### Verwendung des CLI-Tools

```bash
# Text analysieren
python -m ethics_model.api.cli analyze --text "Unternehmen sollten Profit über alles andere stellen."

# Oder aus einer Datei
python -m ethics_model.api.cli analyze --file texte.txt

# Batch-Verarbeitung
python -m ethics_model.api.cli batch --file texte_liste.txt --output ergebnisse.json

# Informationen über moralische Frameworks abrufen
python -m ethics_model.api.cli frameworks

# Visualisierung erstellen
python -m ethics_model.api.cli visualize --text "Unternehmen sollten Profit über alles andere stellen." --type frameworks --output viz.json

# Training starten
python -m ethics_model.api.cli train --file training_data.csv --epochs 5 --output trained_model.pt
```

## API-Endpunkte

- `GET /`: API-Statusinformationen
- `POST /analyze`: Analysieren Sie einen Text
- `POST /analyze/batch`: Analysieren Sie mehrere Texte in einem Batch
- `POST /analyze/async`: Starten Sie eine asynchrone Textanalyse
- `GET /tasks/{task_id}`: Prüfen Sie den Status einer asynchronen Aufgabe
- `GET /frameworks`: Informationen über unterstützte moralische Frameworks
- `GET /manipulation-techniques`: Informationen über erkennbare Manipulationstechniken
- `GET /health`: Gesundheitsstatus der API
- `POST /training/train`: Starten Sie eine asynchrone Modelltraining-Sitzung
- `GET /training/train/{task_id}`: Prüfen Sie den Status einer Trainingsaufgabe
- `GET /training/train/{task_id}/result`: Holen Sie die Ergebnisse eines abgeschlossenen Trainings
- `POST /visualization/visualize`: Erstellen Sie Visualisierungen der Modellanalyse

Vollständige API-Dokumentation ist unter `/docs` verfügbar, wenn der Server läuft.

## Konfigurationsoptionen

| Umgebungsvariable | Beschreibung | Standard |
|-------------------|--------------|----------|
| ETHICS_API_HOST | Host-Adresse | 0.0.0.0 |
| ETHICS_API_PORT | Port | 8000 |
| ETHICS_API_DEBUG | Debug-Modus aktivieren | False |
| ETHICS_API_MODEL_PATH | Pfad zum Modellverzeichnis | None |
| ETHICS_API_CHECKPOINT_PATH | Pfad zum Modell-Checkpoint | None |
| ETHICS_API_TOKENIZER_NAME | Name des zu verwendenden Tokenizers | gpt2 |
| ETHICS_API_LLM_NAME | Name des zu verwendenden LLM | gpt2 |
| ETHICS_API_DEVICE | Gerät für die Modellausführung (cuda/cpu) | Automatisch |
| ETHICS_API_MAX_SEQUENCE_LENGTH | Maximale Sequenzlänge für die Verarbeitung | 512 |

## Beispiele

### cURL

```bash
# Text analysieren
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Unternehmen sollten Profit über alles andere stellen.", "include_details": false}'

# Batch-Verarbeitung
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"], "include_details": false}'
```

### Python Requests

```python
import requests

# Text analysieren
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "Unternehmen sollten Profit über alles andere stellen.",
        "include_details": True
    }
)

result = response.json()
print(result)
```

## Integration mit anderen Diensten

Die API kann leicht in andere Dienste und Workflows integriert werden:

- **Web-Anwendungen**: Verwenden Sie die API für Echtzeit-Ethikanalyse in Web-Apps
- **Content-Moderation**: Automatisieren Sie die Überprüfung von Inhalten auf ethische Bedenken
- **Forschungsanalyse**: Verarbeiten Sie große Textkorpora zur ethischen Analyse
- **Bildungswerkzeuge**: Erstellen Sie Lerntools für ethisches Denken und Erkennen von Manipulationstechniken

## Entwicklung

### Abhängigkeiten

Die API verwendet die folgenden Hauptbibliotheken:

- FastAPI: Für das Web-Framework
- Pydantic: Für Datenvalidierung
- Uvicorn: ASGI-Server
- PyTorch: Für die Modellausführung
- Transformers: Für Tokenisierung und LLM-Embedding

### Best Practices

- **Ladbare Checkpoints**: Verwenden Sie die `ETHICS_API_CHECKPOINT_PATH`-Umgebungsvariable, um benutzerdefinierte Modell-Checkpoints zu laden
- **Caching**: Die API implementiert LRU-Caching für Modelle und Tokenizer
- **Timeouts**: Konfigurieren Sie angemessene Timeouts für lange Anfragen
- **Error Handling**: Ordnungsgemäße Fehlerbehandlung und Logging für robuste Betriebsführung

## Leistungsoptimierung

- **Batch-Verarbeitung**: Verwenden Sie den `/analyze/batch`-Endpunkt für mehrere Texte
- **Asynchrone Verarbeitung**: Verwenden Sie den `/analyze/async`-Endpunkt für lange Analysen
- **GPU-Beschleunigung**: Die API nutzt automatisch CUDA, wenn verfügbar
