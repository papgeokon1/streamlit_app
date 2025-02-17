import os
import asyncio
import urllib.request
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from helper_functions import *
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import streamlit as st

from datasets import load_dataset
# Από τα secrets του Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]

class SimpleRAG:
    def __init__(self, urls=None, pdf_files=None, json_files=None, jsonl_files=None, 
                 html_files=None, csv_files=None, txt_files=None, jpeg_files=None, direct_txt_content="", dataset=None):
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
        self._initialize_pipeline()

    async def load_data(self):
        combined_content = ""
        tasks = []
        
        for url in self.urls:
            tasks.append(fetch_text_from_url(url))
        for json_file in self.json_files:
            tasks.append(fetch_text_from_json(json_file))
        for jsonl_file in self.jsonl_files:
            tasks.append(fetch_text_from_jsonl(jsonl_file))
        for html_file in self.html_files:
            tasks.append(fetch_text_from_html(html_file))
        for csv_file in self.csv_files:
            tasks.append(fetch_text_from_csv(csv_file))
        for txt_file in self.txt_files:
            tasks.append(fetch_text_from_txt(txt_file))
        for jpeg_file in self.jpeg_files:
            tasks.append(fetch_text_from_jpeg(jpeg_file))
        
        contents = await asyncio.gather(*tasks)
        combined_content += "\n".join(filter(None, contents))
        
        for pdf_file in self.pdf_files:
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            combined_content += "\n".join(doc.page_content for doc in pdf_docs)
        
        if self.direct_txt_content:
            combined_content += "\n" + self.direct_txt_content
        
        if self.dataset:
            combined_content += "\n" + "\n".join(self.dataset)
        
        if not combined_content.strip():
            raise ValueError("No content was loaded.")
        
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
        embedder = OpenAITextEmbedder()
        doc["embedding"] = embedder.run({"text": content})["embedding"]
        self.document_store.write_documents([doc])

    def query(self, query):
        result = self.pipeline.run(data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}})
        return result["llm"]["replies"][0]