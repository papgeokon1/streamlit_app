# Streamlit QA System with RAG Architectures

This repository hosts a Streamlit-based application that demonstrates a Question-Answering (QA) system built upon different Retrieval-Augmented Generation (RAG) architectures.

##  Overview

The application allows users to upload documents (e.g. PDFs, text files) or provide URLs, and then pose natural language questions. The system combines several RAG-based approaches to retrieve relevant context from the data and generate informed answers.

Included architectures:
- **Simple RAG**: Basic retrieval + generation pipeline.
- **Self RAG**: Uses feedback from the model itself to refine queries.
- **Graph RAG**: Constructs a knowledge graph from the documents and uses it to enhance contextual understanding.

##  Features

-  Upload and analyze custom documents (PDFs, JSONL, etc.)
-  Fetch content from web pages
-  Ask questions and get answers via:
  - Simple RAG 
  - Self RAG
  - Graph-based RAG
-  Visualize the knowledge graph (Graph RAG)
-  Keyword extraction and memory monitoring

## 📂 Project Structure

streamlit_app/ │ ├── .devcontainer/ # Development container configuration │ ├── data/ # Directory for storing example or uploaded data │ ├── evaluate_rag.py # Script to evaluate and compare RAG approaches ├── graph_rag_v2.py # Graph RAG implementation using knowledge graphs ├── helper_functions.py # Shared utility functions for fetching and parsing ├── keyword_analysis.py # Performs keyword/topic extraction from documents ├── memory_monitor.py # Optional tool to monitor memory usage ├── self_rag.py # Implementation of Self RAG logic ├── simple_rag_haystack.py # Basic RAG pipeline │ ├── main_app.py # Streamlit application  ├── requirements.txt # List of required Python packages └── README.md #