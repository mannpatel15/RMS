#!/bin/bash

# Navigate to the script's directory to ensure paths are correct
# This makes the script runnable from anywhere
cd "$(dirname "$0")"

# Define the log file path
LOG_FILE="./data/monitor_output.log"

# --- Log Trimming Logic ---
# If the log file exists, trim it to the last 100 lines.
# A temp file is used to do this safely.
if [ -f "$LOG_FILE" ]; then
    tail -n 100 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

# --- Run the Monitor ---
# Run the python script and append (>>) both standard output and standard error to the log file.
# The '2>&1' part ensures that error messages are also captured in the log.
/usr/local/bin/python3 ./src/monitor_online_adaptive.py >> "$LOG_FILE" 2>&1