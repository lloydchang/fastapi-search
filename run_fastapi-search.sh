#!/bin/sh -x

# File: run_fastapi-serach.sh

uvicorn api.index:app --reload
