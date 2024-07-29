#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "myenv" ]; then
    python -m venv myenv
fi

# Activate the virtual environment
source myenv/bin/activate

# Print the virtual environment path to verify
echo "Virtual environment activated: $VIRTUAL_ENV"
