#!/usr/bin/env bash

# Install Tesseract
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# Install Python dependencies
pip install -r requirements.txt
