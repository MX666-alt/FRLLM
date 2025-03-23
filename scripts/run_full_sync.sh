#!/bin/bash

# Dieser Script führt eine vollständige manuelle Synchronisierung durch
# und zeigt detaillierte Ausgaben an

echo "Starte vollständige Dropbox-Synchronisierung..."
echo "---------------------------------------------"

# Zum Projektverzeichnis wechseln
cd /opt/immobilien-rag

# Virtuelle Umgebung aktivieren
source venv/bin/activate

# Führe das Synchronisierungsskript aus
python /opt/immobilien-rag/scripts/dropbox_sync.py

# Zeige die letzten 20 Zeilen des Logs an
echo ""
echo "Letzte Logeinträge:"
echo "---------------------------------------------"
tail -n 20 /opt/immobilien-rag/logs/dropbox_sync.log

# Zeige den Status der indexierten Dokumente an
echo ""
echo "Status der indexierten Dokumente:"
echo "---------------------------------------------"
cat /opt/immobilien-rag/data/sync_status.json | python -m json.tool

echo ""
echo "Synchronisierung abgeschlossen."
