#!/bin/bash
# 用法: run_jobs.sh <script.py> <jobs.txt> <并行数>
cd "$(dirname "$0")"
xargs -P "$3" -L 1 ../.venv/bin/python -W ignore "$1" < "$2"
