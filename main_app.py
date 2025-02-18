import streamlit as st
from io import BytesIO
import tempfile
import asyncio
import os
from datasets import load_dataset
from self_rag import SelfRAG
from graph_rag_v2 import GraphRAG
from keyword_analysis import find_common_keywords
from simple_rag_haystack import SimpleRAG
from memory_monitor import check_memory_usage
import time

# Function to run async functions synchronously
def run_async(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    return result

# Streamlit App Interface
st.title("QA Assistant with RAG systems")

# Load dataset function
@st.cache_data
def load_orthopedics():
    return load_dataset("caleboh/tka_tha_meta_analysis", encoding="ISO-8859-1")

# Function to clean dataset
def extract_field_from_dataset(dataset, field=None):
    extracted_texts = []
    for entry in dataset:
        if isinstance(entry, dict):
            if field and field in entry and isinstance(entry[field], str):
                extracted_texts.append(entry[field].strip())
            elif not field:
                extracted_texts.append(" ".join(str(value).strip() for value in entry.values() if isinstance(value, str)))
    return extracted_texts

# Checkbox for dataset usage
use_dataset = st.checkbox("Use Preloaded Dataset")
dataset_field = None
dataset = None

if use_dataset:
    st.write("Loading dataset...")
    raw_dataset = load_orthopedics()
    dataset_split = raw_dataset["train"]

    all_fields = list(dataset_split.features.keys())
    dataset_field = st.selectbox("Select a field to extract (optional)", ["All Fields"] + all_fields)

    if dataset_field == "All Fields":
        cleaned_dataset = extract_field_from_dataset(dataset_split)
    else:
        cleaned_dataset = extract_field_from_dataset(dataset_split, field=dataset_field)

    st.success("Dataset loaded and cleaned successfully!")

# File uploaders for different formats
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
uploaded_jsonls = st.file_uploader("Upload JSONL files", type="jsonl", accept_multiple_files=True)
uploaded_jsons = st.file_uploader("Upload JSON files", type="json", accept_multiple_files=True)
uploaded_htmls = st.file_uploader("Upload HTML files", type="html", accept_multiple_files=True)
uploaded_csvs = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
uploaded_txts = st.file_uploader("Upload TXT files", type="txt", accept_multiple_files=True)
direct_txt_content = st.text_area("Write your TXT content here:")
url_input = st.text_area("Enter URLs (one per line)", "")

# User query input
query = st.text_input("Enter your query:")

if st.button("Get Answer with Simple RAG"):
    dataset = cleaned_dataset if use_dataset else None
    pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, urls = [], [], [], [], [], [], []
    urls = url_input.splitlines() if url_input else []
    
    simple_rag = SimpleRAG(
        urls=urls,
        pdf_files=pdf_files,
        json_files=json_files,
        jsonl_files=jsonl_files,
        html_files=html_files,
        csv_files=csv_files,
        txt_files=txt_files,
        direct_txt_content=direct_txt_content,
        dataset=dataset
    )
    simple_response = simple_rag.run(query)
    st.write(f"Simple RAG Response: {simple_response}")
    
    if st.button("More Detailed Answer (Graph RAG)"):
        graph_rag = GraphRAG(
            urls=urls,
            pdf_files=pdf_files,
            json_files=json_files,
            jsonl_files=jsonl_files,
            html_files=html_files,
            csv_files=csv_files,
            txt_files=txt_files,
            direct_txt_content=direct_txt_content,
            dataset=dataset
        )
        asyncio.run(graph_rag.initialize())
        detailed_response, subgraph = graph_rag.query(query)
        st.write("Graph RAG Answer:")
        st.write(detailed_response)
        if subgraph:
            st.write("Subgraph Visualization:")
            graph_rag.visualizer.visualize_subgraph(subgraph)
            st.write("Node Contents:")
            for node_id, data in subgraph.nodes(data=True):
                node_label = data.get("label", f"Node {node_id}")
                with st.expander(f"{node_label} Content"):
                    content = graph_rag.visualizer.display_node_content(subgraph, node_id)
                    st.write(content)

    if st.button("More Accurate Answer (Self RAG)"):
        self_rag = SelfRAG(
            urls=urls,
            pdf_files=pdf_files,
            json_files=json_files,
            jsonl_files=jsonl_files,
            html_files=html_files,
            csv_files=csv_files,
            txt_files=txt_files,
            direct_txt_content=direct_txt_content,
            dataset=dataset
        )
        accurate_response = self_rag.run(query)
        st.write(f"Self RAG Response: {accurate_response}")

# Sidebar Memory Monitoring
st.sidebar.title("Memory Usage Statistics")
memory_placeholder = st.sidebar.empty()
while True:
    memory_placeholder.text(check_memory_usage())
    time.sleep(2)
