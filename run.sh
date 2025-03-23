#!/bin/bash

# Aktiviere die virtuelle Umgebung
source /opt/immobilien-rag/venv/bin/activate

# Starte die Anwendung
cd /opt/immobilien-rag
python -m app.main
