#!/bin/bash
python -m spacy download en_core_web_sm
exec gunicorn python.app:app
