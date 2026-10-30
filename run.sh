#!/bin/bash
cd "$(dirname "$0")"
exec ./.venv/bin/python instant_transmission.py "$@"
