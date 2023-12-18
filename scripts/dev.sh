#!/bin/sh

uvicorn asgi:app --host "127.0.0.1" --port 8000 --log-level "debug" --reload --workers 1