# Streamlit QA System with RAG Architectures

This repository hosts a **Streamlit-based Question Answering (QA) system** built upon multiple **Retrieval-Augmented Generation (RAG) architectures**.

The project was developed as part of my **Diploma Thesis** in Electrical and Computer Engineering at **Aristotle University of Thessaloniki (AUTH)**, focusing on the **design, implementation, and comparison of RAG-based AI assistants for domain-specific knowledge**.

---

## 📌 Overview

Large Language Models (LLMs) often struggle with hallucinations and lack reliable access to domain-specific or private knowledge.  
Retrieval-Augmented Generation (RAG) addresses this limitation by combining **information retrieval** with **text generation**.

This application allows users to:
- Upload custom documents (e.g. PDFs, JSONL, text files) or provide URLs
- Ask natural language questions
- Select different RAG architectures and compare their behavior through an interactive interface

---

## 🎯 Motivation

Standard LLM-based QA systems rely solely on their internal knowledge and may generate incorrect or unsupported answers.  
This project investigates how different **RAG architectures** improve:
- Factual grounding
- Context relevance
- Reliability of responses over custom document collections

By providing an interactive comparison environment, the system enables empirical observation of architectural differences.

---

## 🧠 Implemented RAG Architectures

The following RAG approaches are implemented and compared:

- **Simple RAG**  
  A baseline retrieval + generation pipeline using vector-based retrieval.

- **Self RAG**  
  Incorporates feedback or self-reflection mechanisms from the model to refine queries and answers.

- **Graph RAG**  
  Constructs a knowledge graph from the ingested documents and leverages graph-based retrieval to enhance contextual understanding and reasoning.

Each architecture follows a different retrieval and reasoning strategy, allowing qualitative comparison of response behavior.

---

## 🔄 System Pipeline

1. User uploads documents or provides URLs  
2. Documents are parsed and indexed  
3. Relevant context is retrieved using the selected RAG strategy  
4. Retrieved context is passed to the LLM for answer generation  
5. The generated answer is displayed via the Streamlit interface  

---

## 🚀 Features

- Upload and analyze custom documents (PDFs, JSONL, text files)
- Fetch and process content from web pages
- Ask questions using:
  - Simple RAG
  - Self RAG
  - Graph-based RAG
- Knowledge graph visualization (Graph RAG)
- Keyword extraction and topic analysis
- Memory and session monitoring utilities

---


## 📂 Project Structure

streamlit_app/
├── .devcontainer/ # Development container configuration
├── data/ # Uploaded or example documents
├── evaluate_rag.py # Evaluation and comparison utilities
├── graph_rag_v2.py # Graph RAG implementation
├── helper_functions.py # Shared utilities for fetching and parsing
├── keyword_analysis.py # Keyword and topic extraction
├── memory_monitor.py # Memory monitoring utilities
├── self_rag.py # Self RAG implementation
├── simple_rag_haystack.py # Baseline RAG pipeline
├── main_app.py # Streamlit application entry point
├── requirements.txt # Python dependencies
└── README.md
