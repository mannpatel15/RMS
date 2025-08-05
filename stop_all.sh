#!/bin/bash

launchctl unload ~/Library/LaunchAgents/com.cua.mastercapture.plist 2>/dev/null
pkill -f data_capture.py
pkill -f main_capture.py

echo "✅ All CUA processes stopped."
