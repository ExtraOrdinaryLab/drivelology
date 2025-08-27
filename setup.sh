#!/bin/bash

sudo apt update
tmux set-option -g mouse on \; bind -T copy-mode-vi MouseDragEnd1Pane send -X copy-selection-and-cancel

pip install -e .
pip install -r requirements.txt