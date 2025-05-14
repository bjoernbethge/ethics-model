"""
Beispiel für die erweiterte Nutzung der Ethics Model API.

Dieses Skript demonstriert die Verwendung von Visualisierungs- und Trainingsfunktionen
der Ethics Model API für fortgeschrittene Anwendungsfälle.
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ethics_model.api.client import EthicsModelClient

# Client initialisieren
client = EthicsModelClient(base_url="http://localhost:8000")

# Beispieltexte für verschiedene Analysen
EXAMPLE_TEXTS = {
    "profit": "Unternehmen sollten Profit über alles andere stellen, ungeachtet ethischer Bedenken.",
    "utilitarian": "Wir sollten immer die Handlung wählen, die das größte Glück für die größte Anzahl von Menschen erzeugt.",
    "deontological": "Es ist unsere Pflicht, die Wahrheit zu sagen, unabhängig von den Konsequenzen.",
    "virtue": "Ein tugendhafter Mensch handelt aus Charakter heraus und nicht aus Kalkül.",
    "manipulative": "Jeder, der das nicht unterstützt, ist entweder dumm oder böswillig - es gibt keine andere Möglichkeit."
}


def test_visualizations():
    """Testen der Visualisierungsfunktionen."""
    print("\n=== Visualisierungsfunktionen testen ===")
    
    # Für jeden Text und Visualisierungstyp eine Visualisierung erstellen
    for text_name, text in EXAMPLE_TEXTS.items():
        print(f"\nAnalysiere Text: {text_name}")
        
        # Framework-Visualisierung
        frameworks_viz = client.visualize(text, visualization_type="frameworks")
        print(f"  Framework-Visualisierung erstellt: {len(frameworks_viz['visualization_data'])} Datenpunkte")
        
        # Manipulations-Visualisierung
        manip_viz = client.visualize(text, visualization_type="manipulation")
        print(f"  Manipulations-Visualisierung erstellt: {len(manip_viz['visualization_data'])} Datenpunkte")
        
        # Aufmerksamkeits-Visualisierung (nur für den ersten Text detailliert anzeigen)
        if text_name == "manipulative":
            attention_viz = client.visualize(text, visualization_type="attention")
            
            # Plotly-Visualisierung mit den Daten erstellen
            print("\nAttention-Heatmap wird angezeigt (schließen Sie das Fenster, um fortzufahren)...")
            
            # Plotly-Konfiguration extrahieren
            plot_config = attention_viz["plot_config"]
            
            # Figur erstellen
            fig = go.Figure(data=plot_config["data"], layout=plot_config["layout"])
            
            # Anzeigen (oder als HTML-Datei speichern)
            fig.write_html("attention_heatmap.html")
            print("  Attention-Heatmap als 'attention_heatmap.html' gespeichert.")


def test_training():
    """Testen der Trainingsfunktionen."""
    print("\n=== Trainingsfunktionen testen ===")
    
    # Einfachen Trainingsdatensatz erstellen
    train_texts = list(EXAMPLE_TEXTS.values())
    
    # Einfache Labels für Demonstration
    ethics_labels = [0.2, 0.8, 0.7, 0.9, 0.1]  # Niedrig für 'profit' und 'manipulative'
    manipulation_labels = [0.7, 0.2, 0.3, 0.1, 0.9]  # Hoch für 'profit' und 'manipulative'
    
    print(f"Starte Training mit {len(train_texts)} Beispielen...")
    
    # Training starten mit minimalen Parametern (für schnelle Demo)
    task_id = client.train(
        train_texts=train_texts,
        ethics_labels=ethics_labels,
        manipulation_labels=manipulation_labels,
        epochs=2,  # Nur wenige Epochen für die Demo
        batch_size=2,
        learning_rate=1e-4,
        checkpoint_name="demo_model"
    )
    
    print(f"Training gestartet. Task-ID: {task_id}")
    
    # Auf Abschluss des Trainings warten
    status = client.get_training_status(task_id)
    
    while status["status"] not in ["completed", "failed"]:
        progress = status["progress"] * 100 if status["progress"] is not None else 0
        print(f"Fortschritt: {progress:.1f}% (Epoche {status['current_epoch']}/{status['total_epochs']})")
        
        if status["train_loss"] is not None:
            print(f"  Train Loss: {status['train_loss']:.4f}")
        if status["val_loss"] is not None:
            print(f"  Validation Loss: {status['val_loss']:.4f}")
        
        # Kurz warten und dann Status aktualisieren
        time.sleep(2)
        status = client.get_training_status(task_id)
    
    if status["status"] == "completed":
        print("Training erfolgreich abgeschlossen!")
        
        # Ergebnisse abrufen
        result = client.get_training_result(task_id)
        
        # Ergebnisse anzeigen
        print("\n=== Trainingsergebnisse ===")
        print(f"Abgeschlossene Epochen: {result['epochs_completed']}")
        print(f"Training-Ethik-Genauigkeit: {result['train_ethics_accuracy']:.4f}")
        print(f"Training-Manipulations-Genauigkeit: {result['train_manipulation_accuracy']:.4f}")
        
        if result["val_ethics_accuracy"] is not None:
            print(f"Validierungs-Ethik-Genauigkeit: {result['val_ethics_accuracy']:.4f}")
        if result["val_manipulation_accuracy"] is not None:
            print(f"Validierungs-Manipulations-Genauigkeit: {result['val_manipulation_accuracy']:.4f}")
        
        print(f"Trainingsdauer: {result['training_duration_seconds']:.1f} Sekunden")
        
        if result["checkpoint_path"]:
            print(f"Modell-Checkpoint gespeichert unter: {result['checkpoint_path']}")
        
        # Verlaufskurve zeichnen
        if result["train_loss_history"] and len(result["train_loss_history"]) > 0:
            epochs = list(range(1, len(result["train_loss_history"]) + 1))
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, result["train_loss_history"], 'b-', label='Training Loss')
            
            if result["val_loss_history"] and len(result["val_loss_history"]) > 0:
                plt.plot(epochs, result["val_loss_history"], 'r-', label='Validation Loss')
            
            plt.title('Trainingsverlauf')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_history.png')
            print("Trainingsverlauf als 'training_history.png' gespeichert.")
    else:
        print(f"Training fehlgeschlagen: {status.get('error', 'Unbekannter Fehler')}")


def test_inference_with_trained_model():
    """Testen der Inferenz mit dem trainierten Modell."""
    print("\n=== Inferenz mit trainiertem Modell testen ===")
    
    # Neuer Text zum Testen
    test_text = "Es ist völlig absurd zu behaupten, dass Unternehmen eine soziale Verantwortung haben, die über Gewinnmaximierung hinausgeht."
    
    # Analyse durchführen
    print(f"Analysiere Text: {test_text}")
    result = client.analyze(test_text, include_details=True)
    
    # Ergebnisse anzeigen
    print("\n=== Analyseergebnisse ===")
    print(f"Ethik-Score: {result['ethics_score']:.4f}")
    print(f"Manipulations-Score: {result['manipulation_score']:.4f}")
    print(f"Dominantes Framework: {result['dominant_framework']}")
    
    # Details
    print("\nZusammenfassung:")
    for key, value in result["summary"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # API-Verfügbarkeit prüfen
    if not client.ping():
        print("Fehler: API ist nicht verfügbar. Bitte starten Sie den API-Server.")
        exit(1)
    
    try:
        # Funktionen testen
        test_visualizations()
        test_training()
        test_inference_with_trained_model()
        
        print("\nAlle Tests erfolgreich abgeschlossen!")
    
    except Exception as e:
        print(f"Fehler: {str(e)}")
