from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError
from typing import List
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from openai import RateLimitError
from rank_bm25 import BM25Okapi
import numpy as np

import asyncio
import random
import textwrap
import numpy as np
import requests
from bs4 import BeautifulSoup

import requests

from io import BytesIO

import aiohttp
import aiofiles
import json
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

async def fetch_text_from_jpeg(jpeg_file):
    """
    Εξάγει κείμενο από ένα αρχείο JPEG χρησιμοποιώντας OCR.
    """
    try:
        # Άνοιγμα της εικόνας
        image = Image.open(jpeg_file)
        
        # Εξαγωγή κειμένου με OCR
        text = pytesseract.image_to_string(image)
        
        return text.strip()
    except Exception as e:
        print(f"Error processing JPEG file: {e}")
        return ""



async def fetch_text_from_json(json_file):
    """
    Εξάγει περιεχόμενο από ένα JSON αρχείο και το επιστρέφει ως string.
    """
    async with aiofiles.open(json_file, 'r') as file:
        content = await file.read()
    data = json.loads(content)
    return json.dumps(data, indent=4)

async def fetch_text_from_jsonl(jsonl_file):
    """
    Εξάγει περιεχόμενο από ένα JSONL αρχείο και το επιστρέφει ως string.
    """
    content = ""
    async with aiofiles.open(jsonl_file, 'r') as file:
        async for line in file:
            data = json.loads(line)
            content += json.dumps(data) + "\n"
    return content

async def fetch_text_from_txt(txt_file):
    """
    Εξάγει το περιεχόμενο από ένα TXT αρχείο και το επιστρέφει ως string.
    """
    async with aiofiles.open(txt_file, 'r') as file:
        content = await file.read()
    return content

async def fetch_text_from_csv(csv_file):
    """
    Εξάγει το περιεχόμενο από ένα CSV αρχείο και το επιστρέφει ως string.
    """
    df = pd.read_csv(csv_file)
    return df.to_string(index=False)

async def fetch_text_from_html(html_file):
    """
    Εξάγει το καθαρό κείμενο από ένα HTML αρχείο και το επιστρέφει ως string.
    """
    async with aiofiles.open(html_file, 'r') as file:
        content = await file.read()
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator="\n")
    clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return clean_text


async def fetch_text_from_url(url):
    """
    Ασύγχρονη συνάρτηση που εξάγει το κείμενο από ένα URL και το επιστρέφει ως string.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()  # Αναφέρει σφάλμα για κακές καταστάσεις
                html_content = await response.text()

                # Χρησιμοποιεί το BeautifulSoup για ανάλυση του HTML
                soup = BeautifulSoup(html_content, "html.parser")

                # Εξαγωγή του κειμένου από τη σελίδα
                text = soup.get_text(separator="\n")

                # Καθαρισμός του κειμένου
                clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

                return clean_text

    except aiohttp.ClientError as e:
        print(f"Error fetching the URL: {e}")
        return None


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_url(url, chunk_size=1000, chunk_overlap=200):
    """
    Encodes text from a URL into chunks using OpenAI embeddings.
    
    Args:
        url: The URL of the webpage to fetch.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.
    
    Returns:
        A list of chunks (documents) created from the URL content.
    """
    # Fetch text from the URL
    content = fetch_text_from_url(url)
    if not content:
        raise ValueError("Failed to retrieve or process the URL content.")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.create_documents([content])

    return chunks  # Επιστρέφουμε τα chunks εδώ






def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a string into a vector store using OpenAI embeddings.

    Args:
        content (str): The text content to be encoded.
        chunk_size (int): The size of each chunk of text.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        FAISS: A vector store containing the encoded content.

    Raises:
        ValueError: If the input content is not valid.
        RuntimeError: If there is an error during the encoding process.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        # Generate embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context and URLs.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    """

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]

    return context


class QuestionAnswerFromContext(BaseModel):
    """
    Model to generate an answer to a query based on a given context.
    
    Attributes:
        answer_based_on_content (str): The generated answer based on the context.
    """
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm):
    # Initialize the ChatOpenAI model with specific parameters
    question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(
        QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Answer a question using the given context by invoking a chain of reasoning.

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {
        "question": question,
        "context": context
    }
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        #print("/n")




def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    Perform BM25 retrieval and return the top k cleaned text chunks.

    Args:
    bm25 (BM25Okapi): Pre-computed BM25 index.
    cleaned_texts (List[str]): List of cleaned text chunks corresponding to the BM25 index.
    query (str): The query string.
    k (int): The number of text chunks to retrieve.

    Returns:
    List[str]: The top k cleaned text chunks based on BM25 scores.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top k scores
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Retrieve the top k cleaned text chunks
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


async def exponential_backoff(attempt):
    """
    Implements exponential backoff with a jitter.
    
    Args:
        attempt: The current retry attempt number.
        
    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")

    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.
    
    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.
        
    Returns:
        The result of the coroutine if successful.
        
    Raises:
        The last encountered exception if all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except RateLimitError as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e

            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)

    # If max retries are reached without success, raise an exception
    raise Exception("Max retries reached")