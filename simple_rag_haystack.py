import os
import sys
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import PDFMinerLoader
import asyncio
import streamlit as st
from datasets import load_dataset
# Από τα secrets του Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]


sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks
from helper_functions import *
from evaluate_rag import *


# Prompt για την απάντηση
response_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)


class SimpleRAG:
    def __init__(self, urls=None, pdf_files=None, json_files=None, jsonl_files=None, html_files=None, csv_files=None, txt_files=None, direct_txt_content="", dataset=None, top_k=1):
        combined_parts = []
        tasks = []

        if urls:
            for url in urls:
                tasks.append(fetch_text_from_url(url))

            results = asyncio.run(asyncio.gather(*tasks))

            for res in results:
                if res:
                    combined_parts.append(res)

        combined_content = "\n\n".join(combined_parts)

        if not combined_content.strip():
            raise ValueError("The combined content is empty. Cannot create vectorstore.")

        # Επεξεργασία JSON αρχείων
        if json_files:
            for json_path in json_files:
                tasks.append(fetch_text_from_json(json_path))

        # Επεξεργασία JSONL αρχείων
        if jsonl_files:
            for jsonl_path in jsonl_files:
                tasks.append(fetch_text_from_jsonl(jsonl_path))

        # Επεξεργασία HTML αρχείων
        if html_files:
            for html_path in html_files:
                tasks.append(fetch_text_from_html(html_path))

        # Επεξεργασία CSV αρχείων
        if csv_files:
            for csv_path in csv_files:
                tasks.append(fetch_text_from_csv(csv_path))

        # Επεξεργασία TXT αρχείων
        if txt_files:
            for txt_path in txt_files:
                tasks.append(fetch_text_from_txt(txt_path))

        # Προσθήκη περιεχομένου από το text area
        if direct_txt_content:
            combined_content += direct_txt_content + "\n\n"

        # Εκτέλεση όλων των εργασιών παράλληλα και συνένωση των αποτελεσμάτων
        contents = asyncio.run(self._fetch_all_contents(tasks))
        for content in contents:
            if content:
                combined_content += content + "\n\n"

        if pdf_files:
            for pdf_path in pdf_files:
                print(f"Processing PDF file: {pdf_path}")
                try:
                    pdf_text = extract_text_from_pdf_pymupdf(pdf_path)
                    if pdf_text:
                        combined_content += f"\n--- Content from {os.path.basename(pdf_path)} ---\n"
                        combined_content += pdf_text + "\n\n"
                except Exception as e:
                    print(f"Error parsing {pdf_path}: {e}")

        # Προσθήκη περιεχομένου από dataset
        if dataset:
            combined_content += "\n\n".join(dataset) + "\n\n"

        # Δημιουργία vectorstore
        if combined_content.strip():
            self.vectorstore = encode_from_string(combined_content)
        else:
            raise ValueError("The combined content is empty. Cannot create vectorstore.")

        # Αρχικοποίηση LLM
        self.top_k = top_k
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
        self.response_chain = response_prompt | self.llm

    async def _fetch_all_contents(self, tasks):
        """Εκτέλεση όλων των ασύγχρονων εργασιών."""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if isinstance(result, str)]
    
    def run(self, query):
        print(f"Processing query: {query}")

        # Ανάκτηση σχετικών εγγράφων
        print("Retrieving relevant documents...")
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        contexts = [doc.page_content for doc in docs]

        # Αν δεν υπάρχουν έγγραφα, απαντάει χωρίς context
        if not contexts:
            print("No relevant documents found. Generating response without retrieval...")
            input_data = {"query": query, "context": ""}
            return self.response_chain.invoke(input_data).response

        # Δημιουργία απάντησης με βάση τα ανακτημένα έγγραφα
        print("Generating response...")
        input_data = {"query": query, "context": "\n".join(contexts)}
        return self.response_chain.invoke(input_data).content


