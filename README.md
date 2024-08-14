# WebScraper Tool

**Status:** _In Entwicklung_

## Beschreibung

Dieses Projekt ist ein fortschrittlicher Webscraper, der darauf ausgelegt ist, automatisch Impressum-Seiten von Websites zu finden und relevante Informationen wie den Firmennamen, den Kontakt und die E-Mail-Adresse zu extrahieren. Darüber hinaus wird ein NLP-Modell (Natural Language Processing) verwendet, um diese Informationen genauer zu identifizieren. Das Projekt befindet sich derzeit in der Entwicklungsphase, und das LLM (Large Language Model) wird aktiv trainiert, um die Extraktionsergebnisse zu verbessern.

## Hauptfunktionen

- **Automatisches Auffinden von Impressum-Seiten:** Der Scraper durchsucht Webseiten nach Links, die wahrscheinlich auf Impressum-Seiten verweisen.
- **Extraktion von Firmennamen, Kontaktpersonen und E-Mail-Adressen:** Mithilfe von regulären Ausdrücken und NLP-Methoden werden relevante Informationen auf der Impressum-Seite identifiziert und extrahiert.
- **Training des NLP-Modells:** Das NLP-Modell wird kontinuierlich mit spezifischen Trainingsdaten trainiert, um die Genauigkeit der Informationen zu verbessern.

## Installation

Zur Nutzung dieses Tools sind folgende Schritte erforderlich:

1. **Python-Umgebung einrichten:**
   - Stellen Sie sicher, dass Python 3.x installiert ist.
   - Installieren Sie die benötigten Bibliotheken mit dem folgenden Befehl:
     ```bash
     pip install -r requirements.txt
     ```

2. **NLP-Modell herunterladen:**
   - Das Tool nutzt das `de_core_news_sm` Modell von SpaCy. Es wird automatisch heruntergeladen, wenn es nicht bereits vorhanden ist.

## Nutzung

### Hauptskript

Das Hauptskript für den Webscraper ist `webscraper.py`. Die Funktionalitäten umfassen:

- **fetch_url:** Ruft den HTML-Inhalt einer gegebenen URL ab.
- **find_impressum_page:** Durchsucht die Webseite nach möglichen Impressum-Seiten.
- **extract_info_from_impressum:** Extrahiert Firmennamen, Kontaktpersonen und E-Mail-Adressen von der Impressum-Seite.
- **train_nlp_model:** Trainiert das NLP-Modell mit neuen Trainingsdaten, um die Extraktionsergebnisse zu verbessern.

### Beispielaufruf

```python
from webscraper import WebScraper, CompanyInfo

urls = ["https://beispielseite.de"]
scraper = WebScraper()

for url in urls:
    info = scraper.extract_info_from_url(url)
    if info:
        print(info)

Training des NLP-Modells

Das NLP-Modell wird mit spezifischen Texten und deren Annotationen trainiert. Das Training erfolgt durch die train_nlp_model-Methode, welche das Modell weiter verbessert.

Beispiel für Trainingsdaten

training_data = [
    ("Rechtsanwaltskanzlei Schmidt", {"entities": [(0, 26, "ORG")]}),
    ("Dr. Hans Mustermann", {"entities": [(0, 18, "PER")]}),
    ("kanzlei@anwalt-paderborn.de", {"entities": [(0, 26, "EMAIL")]}),
    # Weitere Trainingsbeispiele...
]

scraper.train_nlp_model(training_data)
Lizenz

Dieses Projekt steht unter der MIT-Lizenz.


Bei Fragen oder Problemen wenden Sie sich bitte an stephanbrockmeier6@gmail.com

