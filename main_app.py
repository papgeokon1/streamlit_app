import streamlit as st
from io import BytesIO
import tempfile
import asyncio
import os
import json
from self_rag import SelfRAG
from graph_rag_v2 import GraphRAG
import io
import zipfile

# Function to run async functions synchronously
def run_async(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    return result

# Clear Database function
def clear_database():
    st.session_state.pop('self_rag_instance', None)
    st.session_state.pop('graph_rag_instance', None)
    st.success("Database cleared successfully.")

# Function to load dataset from ZIP
def load_cleaned_dataset_from_zip():
    zip_path = "data/cleaned_lawstack.zip"  # Path to the zip file in your repository
    json_file = "cleaned_lawstack.json"    # The name of the JSON file inside the ZIP

    # Extract the JSON file from the ZIP
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file) as f:
            dataset = json.load(f)

    return dataset

# Streamlit App Interface
st.title("RAG Assistant")

# Choose RAG Model
rag_option = st.selectbox("Choose RAG Model", ("Self RAG", "Graph RAG"))

# Clear Database Button
if st.button("Clear Database"):
    clear_database()

# File uploaders for different formats
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
uploaded_jsonls = st.file_uploader("Upload JSONL files", type="jsonl", accept_multiple_files=True)
uploaded_jsons = st.file_uploader("Upload JSON files", type="json", accept_multiple_files=True)
uploaded_htmls = st.file_uploader("Upload HTML files", type="html", accept_multiple_files=True)
uploaded_csvs = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
uploaded_txts = st.file_uploader("Upload TXT files", type="txt", accept_multiple_files=True)
direct_txt_content = st.text_area("Write your TXT content here:")
url_input = st.text_area("Enter URLs (one per line)", "")

# Checkbox for dataset usage
use_dataset = st.checkbox("Use Preloaded Dataset")
dataset = None


if use_dataset:
    st.write("Loading dataset from ZIP...")
    try:
        # Load dataset from the ZIP file
        cleaned_dataset = load_cleaned_dataset_from_zip()
        st.success("Dataset loaded successfully!")
        st.write(f"Number of entries: {len(cleaned_dataset)}")
        st.write("Preview:")
        st.write(cleaned_dataset[:5])  # Show the first 5 entries
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")

# User query input
query = st.text_input("Enter your query:")

# Function to handle RAG model execution
def run_rag_model(rag_option, urls, pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, direct_txt_content, query, dataset=None):
    if rag_option == "Self RAG":
        rag = SelfRAG(
            urls=urls,
            pdf_files=pdf_files,
            json_files=json_files,
            jsonl_files=jsonl_files,
            html_files=html_files,
            csv_files=csv_files,
            txt_files=txt_files,
            direct_txt_content=direct_txt_content,
            dataset=dataset,
        )
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
            direct_txt_content=direct_txt_content,
            dataset=dataset,
        )
        asyncio.run(graph_rag.initialize())
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

    # Check inputs
    if not (pdf_files or json_files or jsonl_files or html_files or csv_files or txt_files or urls or direct_txt_content or use_dataset) or not query:
        st.error("Please upload files, provide URLs, or enable the dataset, and input a query.")
    else:
        # Run the RAG Model
        run_rag_model(rag_option, urls, pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, direct_txt_content, query, dataset)

    # Clean up temporary files
    for file_path in pdf_files + json_files + jsonl_files + html_files + csv_files + txt_files:
        os.remove(file_path)
