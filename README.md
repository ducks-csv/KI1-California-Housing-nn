# California Housing Neural Network

Dieses Projekt wurde im Rahmen eines Universitätsprojekts als Modulabschlussprüfung für das Modul **Künstliche Intelligenz 1 (KI I)** entwickelt. Es enthält ein Jupyter Notebook zur Vorhersage von Hauspreisen anhand des California Housing Datensatzes.
Dieses Projekt enthält ein Jupyter Notebook zur Vorhersage von Hauspreisen anhand des California Housing Datensatzes.

## Inhalt

- `final_neural_network.ipynb` – Hauptnotebook mit kompletter Pipeline:
  - Laden und Bereinigen der Daten
  - Feature-Engineering inklusive geografischer Distanzmerkmale
  - Training eines neuronalen Netzes
  - Vergleich mit einem linearen Basismodell
  - Finale Leistungsbewertung
- `development_notebooks/` – zusätzliche Explorations- und Entwicklungsnotebooks.

## Methodik & Ergebnisse (Logbuch-Zusammenfassung)

> **Disclaimer:** Die folgenden Punkte fassen die wesentlichen Erkenntnisse aus dem geführten Logbuch zusammen. Detaillierte Analysen, Versuchsauswertungen und theoretische Hintergründe entstammen der vollständigen Projektdokumentation.
Dieses Repository zeigt eine komprimierte Fassung des Projekts. Das Gesamtprojekt (inklusive des vollständigen Logbuchs) wird auf Anfrage gerne zur Verfügung gestellt.

Aus den Untersuchungen ergeben sich folgende Kernaspekte für die Datenverarbeitung und Modellierung:

- **Datenbereinigung & Ausreißerbehandlung:**
  - Maskierung von Cut-off Points bei `MedInc` (15), `HouseAge` (52) und `MedHouseVal` (5).
  - Um Datenverlust durch Trimming (~32 %) oder IQR-Filterung (~24 %) zu vermeiden, wird **Winsorizing** (Begrenzung auf das 99%-Perzentil) eingesetzt.
  - Stark rechtsschiefe Features (z. B. `AveBedrms` mit Skewness 28.92) wurden logarithmiert. Standardisierung erfolgt mittels `StandardScaler`.
- **Feature Engineering:**
  - **Verhältniskennzahlen:** z. B. `Bedrooms Per Room` und `Income Per Occupant` zur Differenzierung von Haushaltsstrukturen.
  - **Geografische Distanzen:** Berechnung der Distanz (mittels Haversine-Formel) zu den drei Großstädten Los Angeles, San Francisco und San Jose.
  - **Clustering:** K-Means-Clustering der Koordinaten (`Ocean Proximity`) mit anschließendem One-Hot-Encoding zur Erfassung nicht-linearer Lageeffekte.
- **Architektur des Neuronalen Netzes:**
  - Trichterförmiges Shallow Network (128 -> 64 -> 32 Neuronen) mit ReLu-Aktivierung in den Hidden Layern und einer linearen Output-Schicht.
  - **Regularisierung & Optimierung:** Einsatz von Dropout (0.2), Batch Normalization und L2-Regularisierung ($\alpha = 0.001$) zur Reduktion von Overfitting.
  - Steuerung über Callbacks: `ReduceLROnPlateau` und `EarlyStopping`.
- **Ergebnisse:**
Das entwickelte Neuronale Netz übertrifft mit einem R2 = 0, 79 die lineare Baseline (R2 = 0, 67) deutlich.
Diese Steigerung resultiert aus der Fähigkeit des Netzes, durch den Einsatz von Aktivierungsfunktionen
und einer Vielzahl von Neuronen (Units) nicht-lineare Zusammenhänge zwischen den Merkmalen abzubilden.
Die Modellperformanz wird maßgeblich durch vier Faktoren bestimmt: die Datenaufbereitung (Winsorizing,
Log-Transformation, Maskierung) , die Netzarchitektur (Layer und Units) , die Regularisierung mittels
Dropout und L2 (α = 0, 001) sowie das Feature Engineering (geografische Distanzen, Ocean Proximity,
Zimmerverhältnisse).

## Voraussetzungen

Empfohlen ist ein Python 3.10+ Umfeld.

Benötigte Python-Pakete:

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn

## Installation

1. Virtuelle Umgebung erstellen:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt