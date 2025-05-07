# Document Parser

A powerful FastAPI service for parsing, analyzing, and embedding documents from various formats (PDF, images, text files) using AI.

## Overview

Document Parser is a microservice that extracts text and content from documents, processes them using AI models, and stores vector embeddings in Pinecone for later retrieval. It supports:

- PDF documents (extracting text and images)
- Image files (using OCR and caption generation)
- Text files (chunking and embedding)

## Features

- **PDF Processing**: Extracts text and images from PDFs
- **Image Analysis**: Uses OCR to extract text from images and generates captions
- **Text Chunking**: Splits documents into optimal chunks for embedding
- **Vector Embeddings**: Creates embeddings using Cohere's embed-v4.0 model
- **S3 Integration**: Stores and retrieves files from AWS S3
- **Vector Database**: Stores embeddings in Pinecone for efficient retrieval

## Requirements

- Python 3.8+
- AWS S3 credentials
- Cohere API key
- OpenAI API key (via OpenRouter)
- Pinecone API key
- Tesseract OCR (for image processing)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   COHERE_API_KEY=your_cohere_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   AWS_ACCESS_KEY=your_aws_access_key
   AWS_SECRET_KEY=your_aws_secret_key
   AWS_REGION=your_aws_region
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```
4. Install Tesseract OCR (for image processing):
   - On macOS: `brew install tesseract`
   - On Ubuntu: `apt-get install tesseract-ocr`
   - On Windows: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. Start the server:

   ```
   uvicorn main:app --reload
   ```

2. The API provides the following endpoint:

   - `POST /api/parse-and-embed`: Parses a document and creates vector embeddings

   Example request:

   ```json
   {
     "object_name": "document.pdf",
     "user_id": "user123"
   }
   ```

## Deployment

The application is configured for deployment on Render using the `render.yaml` file. It will:

- Install dependencies from `requirements.txt`
- Start the FastAPI server with uvicorn

## Architecture

- `main.py`: FastAPI application entry point with API endpoints
- `handlers.py`: Core logic for handling different document types
- `parsers.py`: Implements document parsing functionality
- `model_calls.py`: Handles AI model API calls
- `s3_handler.py`: AWS S3 integration
- `image_handler.py`: Image processing and OCR capabilities
