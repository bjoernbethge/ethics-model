"""
Beispiel für das Training mit der Ethics Model API im gleichen Stil wie train_with_llm.py.

Dieses Skript zeigt, wie die API für das Training eines EthicsModel mit einem LLM 
in ähnlicher Weise wie train_with_llm.py verwendet werden kann.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ethics_model.api.client import EthicsModelClient

# =====================
# 1. Configuration & Logging
# =====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"  # Anpassen, falls die API auf einem anderen Port läuft
CHECKPOINT_NAME = "trained_ethicsmodel"

# =====================
# 2. Data Preparation
# =====================
def prepare_data():
    """Bereitet Daten für das Training vor."""
    logger.info("Lade Datensatz...")
    ds = load_dataset("flozi00/Fineweb2-German-Eduscore-4andMore", split="train[:1000]")
    texts = ds["text"]
    ethics_labels = [float(x) for x in ds["eduscore"]]
    manipulation_labels = [float(x) for x in ds["manipulation_score"]] if "manipulation_score" in ds.column_names else ethics_labels
    
    return texts, ethics_labels, manipulation_labels

# =====================
# 3. Model Configuration
# =====================
def get_model_config():
    """Erstellt die Modellkonfiguration."""
    # Wir verwenden für die API die gleiche Konfiguration wie in train_with_llm.py
    model_config = {
        'd_model': 2048,  # Gemma-Embedding-Größe (anpassen je nach verwendetem LLM)
        'n_layers': 2,
        'n_heads': 8,
        'max_seq_length': 128,
        'activation': 'gelu',
        'use_gnn': False
    }
    
    return model_config

# =====================
# 4. Main
# =====================
def main():
    # Daten vorbereiten
    texts, ethics_labels, manipulation_labels = prepare_data()
    
    # Modellkonfiguration erstellen
    model_config = get_model_config()
    
    # API-Client erstellen
    client = EthicsModelClient(base_url=API_URL)
    
    # Prüfen, ob API verfügbar ist
    if not client.ping():
        logger.error("API ist nicht erreichbar. Bitte starten Sie den API-Server.")
        sys.exit(1)
    
    # Training starten
    logger.info(f"Starte Training mit {len(texts)} Texten...")
    try:
        task_id = client.train(
            train_texts=texts,
            ethics_labels=ethics_labels,
            manipulation_labels=manipulation_labels,
            validation_split=0.2,
            epochs=5,
            batch_size=4,
            learning_rate=2e-5,
            augment=True,
            checkpoint_name=CHECKPOINT_NAME,
            model_config=model_config
        )
        
        logger.info(f"Training gestartet. Task-ID: {task_id}")
        
        # Auf Trainingsabschluss warten
        status = client.get_training_status(task_id)
        logger.info(f"Anfangsstatus: {status['status']}")
        
        while status["status"] not in ["completed", "failed"]:
            time.sleep(5)  # Alle 5 Sekunden den Status überprüfen
            
            status = client.get_training_status(task_id)
            progress = status["progress"] * 100 if status["progress"] is not None else 0
            epoch_info = f"Epoche {status['current_epoch']}/{status['total_epochs']}" if status['current_epoch'] is not None else ""
            
            logger.info(f"Status: {status['status']}, Fortschritt: {progress:.1f}% {epoch_info}")
            
            if status["train_loss"] is not None:
                logger.info(f"Train Loss: {status['train_loss']:.4f}")
            if status["val_loss"] is not None:
                logger.info(f"Validation Loss: {status['val_loss']:.4f}")
        
        # Trainingsergebnisse abrufen
        if status["status"] == "completed":
            results = client.get_training_result(task_id)
            
            logger.info("\n===== Trainingsergebnisse =====")
            logger.info(f"Abgeschlossene Epochen: {results['epochs_completed']}")
            logger.info(f"Training-Ethik-Genauigkeit: {results['train_ethics_accuracy']:.4f}")
            logger.info(f"Training-Manipulations-Genauigkeit: {results['train_manipulation_accuracy']:.4f}")
            
            if results["val_ethics_accuracy"] is not None:
                logger.info(f"Validierungs-Ethik-Genauigkeit: {results['val_ethics_accuracy']:.4f}")
            if results["val_manipulation_accuracy"] is not None:
                logger.info(f"Validierungs-Manipulations-Genauigkeit: {results['val_manipulation_accuracy']:.4f}")
            
            if results["checkpoint_path"]:
                logger.info(f"Modell-Checkpoint gespeichert unter: {results['checkpoint_path']}")
            
            # Jetzt können wir das trainierte Modell für Vorhersagen verwenden
            logger.info("\n===== Testen des trainierten Modells =====")
            test_text = "Es ist völlig absurd zu behaupten, dass Unternehmen eine soziale Verantwortung haben, die über Gewinnmaximierung hinausgeht."
            
            logger.info(f"Analysiere Text: {test_text}")
            prediction = client.analyze(test_text, include_details=False)
            
            logger.info(f"Ethik-Score: {prediction['ethics_score']:.4f}")
            logger.info(f"Manipulations-Score: {prediction['manipulation_score']:.4f}")
            logger.info(f"Dominantes Framework: {prediction['dominant_framework']}")
        else:
            logger.error(f"Training fehlgeschlagen: {status.get('error', 'Unbekannter Fehler')}")
    
    except Exception as e:
        logger.error(f"Fehler beim Training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
