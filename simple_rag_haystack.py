import os
import sys
import asyncio
import urllib.request
import streamlit as st
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from langchain_community.document_loaders import PDFMinerLoader
from helper_functions import *
from datasets import load_dataset

# Από τα secrets του Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]

class SimpleRAG:
    def __init__(self, urls=None, pdf_files=None, json_files=None, jsonl_files=None, 
                 html_files=None, csv_files=None, txt_files=None, jpeg_files=None, direct_txt_content="", dataset=None, top_k=3):
        self.urls = urls or []
        self.pdf_files = pdf_files or []
        self.json_files = json_files or []
        self.jsonl_files = jsonl_files or []
        self.html_files = html_files or []
        self.csv_files = csv_files or []
        self.txt_files = txt_files or []
        self.jpeg_files = jpeg_files or []
        self.direct_txt_content = direct_txt_content
        self.dataset = dataset or []
        self.document_store = InMemoryDocumentStore()
        self.top_k = top_k
        self._initialize_pipeline()
   
    def load_data(self):
        combined_content = ""

        # Λήψη δεδομένων από URLs (συγχρονισμένη κλήση)
        for url in self.urls:
            try:
                content = fetch_text_from_url(url)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error fetching from URL {url}: {e}")

        # Λήψη δεδομένων από JSON
        for json_file in self.json_files:
            try:
                content = fetch_text_from_json(json_file)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error processing JSON {json_file}: {e}")

        # Λήψη δεδομένων από JSONL
        for jsonl_file in self.jsonl_files:
            try:
                content = fetch_text_from_jsonl(jsonl_file)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error processing JSONL {jsonl_file}: {e}")

        # Λήψη δεδομένων από HTML
        for html_file in self.html_files:
            try:
                content = fetch_text_from_html(html_file)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error processing HTML {html_file}: {e}")

        # Λήψη δεδομένων από CSV
        for csv_file in self.csv_files:
            try:
                content = fetch_text_from_csv(csv_file)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error processing CSV {csv_file}: {e}")

        # Λήψη δεδομένων από TXT
        for txt_file in self.txt_files:
            try:
                content = fetch_text_from_txt(txt_file)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error processing TXT {txt_file}: {e}")

        # Λήψη δεδομένων από εικόνες JPEG
        for jpeg_file in self.jpeg_files:
            try:
                content = fetch_text_from_jpeg(jpeg_file)
                if content:
                    combined_content += content + "\n"
            except Exception as e:
                print(f"Error processing JPEG {jpeg_file}: {e}")

        # Λήψη δεδομένων από PDF
        for pdf_file in self.pdf_files:
            try:
                pdf_loader = PDFMinerLoader(pdf_file)
                pdf_docs = pdf_loader.load()
                pdf_text = "\n".join(doc.page_content for doc in pdf_docs)
                combined_content += pdf_text + "\n"
            except Exception as e:
                print(f"Error processing PDF {pdf_file}: {e}")

        # Προσθήκη δεδομένων από το πεδίο text area
        if self.direct_txt_content:
            combined_content += self.direct_txt_content + "\n"

        # Προσθήκη dataset
        if self.dataset:
            combined_content += "\n".join(self.dataset) + "\n"

        # Αν δεν φορτώθηκε τίποτα, ρίξε exception
        if not combined_content.strip():
            raise ValueError("No content was loaded.")

        # Αποθήκευση στο document store
        self._index_documents(combined_content)

    def _initialize_pipeline(self):
        self.text_embedder = OpenAITextEmbedder()
        self.retriever = InMemoryEmbeddingRetriever(self.document_store)
        self.prompt_builder = PromptBuilder(template="""Given these documents, answer the question.\nDocuments:\n{% for doc in documents %}\n{{ doc.content }}\n{% endfor %}\nQuestion: {{query}}\nAnswer:""")
        self.llm = OpenAIGenerator()
        
        self.pipeline = Pipeline()
        self.pipeline.add_component("text_embedder", self.text_embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

    def _index_documents(self, content):
        embedder = OpenAIDocumentEmbedder()
        doc = {"content": content}
        doc["embedding"] = embedder.run({"documents": [{"content": content}]})["documents"][0]["embedding"]
        self.document_store.write_documents([doc])

    def query(self, query):
        result = self.pipeline.run(data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}})
        return result["llm"]["replies"][0]
