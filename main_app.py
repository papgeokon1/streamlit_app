import streamlit as st
from io import BytesIO
import tempfile
import asyncio
import os
import pandas as pd
from datetime import datetime
from self_rag import SelfRAG 
from graph_rag_v2 import GraphRAG 
from datasets import load_dataset


import io

# Function to run async functions synchronously
def run_async(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    return result

# Clear Database function
def clear_database():
    # Logic to reset or clear RAG instances (reset embeddings, clear graphs)
    st.session_state.pop('self_rag_instance', None)
    st.session_state.pop('graph_rag_instance', None)
    st.success("Database cleared successfully.")

# Streamlit App Interface
st.title("RAG Assistant")
# Choose RAG Model
rag_option = st.selectbox("Choose RAG Model", ("Self RAG", "Graph RAG"))

@st.cache_data
def load_law_stackexchange():
    dataset = load_dataset("ymoslem/Law-StackExchange")
    return dataset

# Clear Database Button
if st.button("Clear Database"):
    clear_database()
# Upload PDF files
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Upload JSONL files
uploaded_jsonls = st.file_uploader("Upload JSONL files", type="jsonl", accept_multiple_files=True)

# Upload JSON files
uploaded_jsons = st.file_uploader("Upload JSON files", type="json", accept_multiple_files=True)

# Upload HTML files
uploaded_htmls = st.file_uploader("Upload HTML files", type="html", accept_multiple_files=True)

# Upload CSV files
uploaded_csvs = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Upload TXT files
uploaded_txts = st.file_uploader("Upload TXT files", type="txt", accept_multiple_files=True)

# Directly inputted TXT content
direct_txt_content = st.text_area("Write your TXT content here:")

# Input URLs
url_input = st.text_area("Enter URLs (one per line)", "")

use_dataset = st.checkbox("Use Law-StackExchange Dataset")
if use_dataset:
    st.write("Loading dataset...")
    dataset = load_law_stackexchange()
    st.success("Dataset loaded successfully.")
    st.write(dataset)

# User query input
query = st.text_input("Enter your query:")


# Function to handle RAG model execution
def run_rag_model(rag_option, urls, pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, direct_txt_content, query):
    if rag_option == "Self RAG":
        rag = SelfRAG(urls=urls, pdf_files=pdf_files, json_files=json_files, jsonl_files=jsonl_files, html_files=html_files, csv_files=csv_files, txt_files=txt_files, direct_txt_content=direct_txt_content)
        if dataset:
            dataset_context = " ".join(data['answers'] for data in dataset['train'][:10])
            query += f"\nDataset Context:\n{dataset_context}"
        response = rag.run(query)
        st.write(f"Response: {response}")        
    elif rag_option == "Graph RAG":
        graph_rag = GraphRAG(
            urls=urls,
            pdf_files=pdf_files,
            json_files=json_files,
            jsonl_files=jsonl_files,
            html_files=html_files,
            csv_files=csv_files,
            txt_files=txt_files,
            direct_txt_content=direct_txt_content
        )
        asyncio.run(graph_rag.initialize())
        if dataset:
            dataset_context = " ".join(data['answers'] for data in dataset['train'][:10])
            query += f"\nDataset Context:\n{dataset_context}"
        final_answer, subgraph = graph_rag.query(query)
        st.write("Answer:")
        st.write(final_answer)
        if subgraph:
            st.write("Subgraph Visualization:")
            graph_rag.visualizer.visualize_subgraph(subgraph)
            st.write("Node Contents:")
            for node_id, data in subgraph.nodes(data=True):
                node_label = data.get("label", f"Node {node_id}")
                with st.expander(f"{node_label} Content"):
                    content = graph_rag.visualizer.display_node_content(subgraph, node_id)
                    st.write(content)
        else:
            st.write("No subgraph available.")

# Execution and Profiling
if st.button("Run Query"):
    # Prepare input files and URLs
    pdf_files = []
    json_files = []
    jsonl_files = []
    html_files = []
    csv_files = []
    txt_files = []
    urls = url_input.splitlines() if url_input else []

    # Save uploaded files to temporary storage
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf.read())
                pdf_files.append(temp_pdf.name)

    if uploaded_jsons:
        for json_file in uploaded_jsons:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_json:
                temp_json.write(json_file.read())
                json_files.append(temp_json.name)

    if uploaded_jsonls:
        for jsonl_file in uploaded_jsonls:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_jsonl:
                temp_jsonl.write(jsonl_file.read())
                jsonl_files.append(temp_jsonl.name)

    if uploaded_htmls:
        for html_file in uploaded_htmls:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html:
                temp_html.write(html_file.read())
                html_files.append(temp_html.name)

    if uploaded_csvs:
        for csv_file in uploaded_csvs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
                temp_csv.write(csv_file.read())
                csv_files.append(temp_csv.name)

    if uploaded_txts:
        for txt_file in uploaded_txts:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
                temp_txt.write(txt_file.read())
                txt_files.append(temp_txt.name)

    if not (pdf_files or json_files or jsonl_files or html_files or csv_files or txt_files or urls or direct_txt_content) or not query:
        st.error("Please upload PDF, JSON, JSONL, HTML, CSV, or TXT files, provide URLs, and input a query.")
    else:
        dataset = load_law_stackexchange() if use_dataset else None
        run_rag_model(rag_option, urls, pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, direct_txt_content, query, dataset)

    # Clean up temporary files
    for pdf_path in pdf_files:
        os.remove(pdf_path)
    for json_path in json_files:
        os.remove(json_path)
    for jsonl_path in jsonl_files:
        os.remove(jsonl_path)
    for html_path in html_files:
        os.remove(html_path)
    for csv_path in csv_files:
        os.remove(csv_path)
    for txt_path in txt_files:
        os.remove(txt_path)
