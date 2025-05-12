#!/bin/bash
# uv run bash init.sh 
# Initialization for this task, only need to run once

SCRIPT_HOME="$PIA_HOME/downstream/Clone-detection-BCB-naturalness-attack"

# Grab the tree-sitter submodules
git submodule init
git submodule update

cd "$SCRIPT_HOME/python_parser/parser_folder"
python build.py
cd "$SCRIPT_HOME"
