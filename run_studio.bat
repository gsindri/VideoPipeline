@echo off
cd /d "C:\Users\gsind\Documents\GitHub\VideoPipeline"
call .venv\Scripts\activate
start /min python -m videopipeline.launcher
