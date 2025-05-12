#!/bin/bash
# uv run bash init.sh 
# Initialization for this task, only need to run once

SCRIPT_HOME="$PIA_HOME/downstream/Clone-detection-BCB-naturalness-attack"

cd "$SCRIPT_HOME/python_parser/parser_folder"
bash build.sh
cd "$SCRIPT_HOME"
