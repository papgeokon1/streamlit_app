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

# Streamlit App Interface
st.title("Medical Assistant for Total Joint Replacement")

# Choose RAG Model
rag_option = st.selectbox("Choose RAG Model", ("Self RAG", "Graph RAG", "Simple RAG"))


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

    elif rag_option == "Simple RAG":
        simple_rag = SimpleRAG(
            urls=urls,
            pdf_files=pdf_files,
            json_files=json_files,
            jsonl_files=jsonl_files,
            html_files=html_files,
            csv_files=csv_files,
            txt_files=txt_files,
            direct_txt_content=direct_txt_content,
        )
        asyncio.run(simple_rag.load_data())
        response = simple_rag.query(query)
        st.write(f"Response: {response}")            

if st.button("Analyze Files"):
    dataset = cleaned_dataset if use_dataset else None

    pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, urls = [], [], [], [], [], [], []

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

    urls = url_input.splitlines() if url_input else []

    if not (pdf_files or json_files or jsonl_files or html_files or csv_files or txt_files or urls or direct_txt_content or use_dataset) or not query:
        st.error("Please upload files, provide URLs, or enable the dataset, and input a query.")
    else:
        run_rag_model(
            rag_option, urls, pdf_files, json_files, jsonl_files, html_files, csv_files, txt_files, direct_txt_content, query, dataset
        )

    for file_path in pdf_files + json_files + jsonl_files + html_files + csv_files + txt_files:
        os.remove(file_path)

# Keyword Analysis Section
if st.checkbox("Analyze Common Keywords Between Datasets"):
    if not dataset and not (uploaded_pdfs or uploaded_jsonls or uploaded_jsons or uploaded_htmls or uploaded_csvs or uploaded_txts):
        st.error("Please upload files or use the preloaded dataset for analysis.")
    else:
        datasets_content = []

        # Συγκέντρωση περιεχομένου από τα αρχεία που ανέβηκαν
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                datasets_content.append(fetch_text_from_pdf(pdf))
        if uploaded_jsonls:
            for jsonl in uploaded_jsonls:
                datasets_content.append(fetch_text_from_jsonl(jsonl))
        if uploaded_csvs:
            for csv in uploaded_csvs:
                datasets_content.append(fetch_text_from_csv(csv))
        if dataset:
            datasets_content.extend(cleaned_dataset)

        # Εύρεση κοινών λέξεων-κλειδιών
        results = find_common_keywords(datasets_content, top_n=10)

        # Εμφάνιση αποτελεσμάτων
        st.subheader("Common Keywords Across Datasets")
        st.write(", ".join(results["common_keywords"]))

        st.subheader("Keywords by Dataset")
        for dataset_name, keywords in results["individual_keywords"].items():
            st.write(f"{dataset_name}: {', '.join(keywords)}")

from memory_monitor import check_memory_usage
import time

st.sidebar.title("Στατιστικά Μνήμης")
memory_placeholder = st.sidebar.empty()  # Δεσμεύει χώρο στο Sidebar

# Ανανεώνει την τιμή μνήμης κάθε 2 δευτερόλεπτα
while True:
    memory_placeholder.text(check_memory_usage())
    time.sleep(2)  # Αναμονή 2 δευτερολέπτων πριν την επόμενη ενημέρωση