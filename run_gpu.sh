#!/bin/bash
# Legacy compatibility - redirects to unified environment
echo "Note: Using unified environment (.venv)"
./.venv/bin/python self_discovery.py "$@"
