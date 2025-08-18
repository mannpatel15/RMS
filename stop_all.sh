#!/bin/bash

# launchctl unload ~/Library/LaunchAgents/com.cua.mastercapture.plist 2>/dev/null
pkill -f data_capture.py
pkill -f main_capture.py
launchctl unload ~/Library/LaunchAgents/com.cua.mastercapture.plist
echo "CUA stopped and launchd disabled. It will not restart until you run start_all.sh."

