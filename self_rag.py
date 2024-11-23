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



class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")


class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant",
                          description="Output only 'Relevant' or 'Irrelevant'.")


class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")


class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported",
                          description="Output 'Fully supported', 'Partially supported', or 'No support'.")


class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")


# Define prompt templates
retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)


# Define main class

class SelfRAG:
    def __init__(self, urls, pdf_files, json_files=None, jsonl_files=None, html_files=None, csv_files=None, txt_files=None, direct_txt_content="",dataset_name=None ,top_k=3):
        combined_content = ""
        tasks = []  # Λίστα για αποθήκευση των ασύγχρονων εργασιών

        # Επεξεργασία URLs
        if urls:
            for url in urls:
                print(f"Fetching content from: {url}")
                tasks.append(fetch_text_from_url(url))  # Προσθέτουμε την εργασία στη λίστα

        # Επεξεργασία JSON αρχείων
        if json_files:
            for json_path in json_files:
                print(f"Processing JSON file: {json_path}")
                tasks.append(fetch_text_from_json(json_path))  # Προσθέτουμε την εργασία στη λίστα

        # Επεξεργασία JSONL αρχείων
        if jsonl_files:
            for jsonl_path in jsonl_files:
                print(f"Processing JSONL file: {jsonl_path}")
                tasks.append(fetch_text_from_jsonl(jsonl_path))  # Προσθέτουμε την εργασία στη λίστα

        # Επεξεργασία HTML αρχείων
        if html_files:
            for html_path in html_files:
                print(f"Processing HTML file: {html_path}")
                tasks.append(fetch_text_from_html(html_path))  # Προσθέτουμε την εργασία στη λίστα

        # Επεξεργασία CSV αρχείων
        if csv_files:
            for csv_path in csv_files:
                print(f"Processing CSV file: {csv_path}")
                tasks.append(fetch_text_from_csv(csv_path))  # Προσθέτουμε την εργασία στη λίστα

        # Επεξεργασία TXT αρχείων
        if txt_files:
            for txt_path in txt_files:
                print(f"Processing TXT file: {txt_path}")
                tasks.append(fetch_text_from_txt(txt_path))  # Προσθέτουμε την εργασία στη λίστα

        # Προσθήκη περιεχομένου από το text area
        if direct_txt_content:
            combined_content += direct_txt_content + "\n\n"

        # Εκτέλεση όλων των εργασιών παράλληλα και συνένωση των αποτελεσμάτων
        contents = asyncio.run(self._fetch_all_contents(tasks))
        
        for content in contents:
            if content:
                combined_content += content + "\n\n"

        # Επεξεργασία PDF αρχείων (συγχρονισμένη κλήση λόγω PDFMinerLoader)
        if pdf_files:
            for pdf_path in pdf_files:
                print(f"Processing PDF file: {pdf_path}")
                pdf_loader = PDFMinerLoader(pdf_path)
                pdf_docs = pdf_loader.load()
                
                for pdf_doc in pdf_docs:
                    combined_content += pdf_doc.page_content + "\n\n"
        # Προσθήκη περιεχομένου από το dataset
        if dataset_name:
            print(f"Loading dataset: {dataset_name}")
            dataset = self.load_dataset(dataset_name)
            for data in dataset['train']:  # Υποθέτουμε ότι έχει split 'train'
                if 'text' in data and data['text'].strip():
                    combined_content += data['text'].strip() + "\n\n"
        # Δημιουργία του vectorstore αν υπάρχει περιεχόμενο
        
        if combined_content:
            self.vectorstore = encode_from_string(combined_content)
        else:
            raise ValueError("Failed to retrieve or process the content from URLs or PDFs.")

        # Αρχικοποίηση άλλων παραμέτρων
        self.top_k = top_k
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

        # Δημιουργία LLMChains για κάθε βήμα, όπως πριν
        self.retrieval_chain = retrieval_prompt | self.llm.with_structured_output(RetrievalResponse)
        self.relevance_chain = relevance_prompt | self.llm.with_structured_output(RelevanceResponse)
        self.generation_chain = generation_prompt | self.llm.with_structured_output(GenerationResponse)
        self.support_chain = support_prompt | self.llm.with_structured_output(SupportResponse)
        self.utility_chain = utility_prompt | self.llm.with_structured_output(UtilityResponse)

    async def _fetch_all_contents(self, tasks):
        """
        Εκτέλεση όλων των ασύγχρονων εργασιών.
        """
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if isinstance(result, str)]
    
    def run(self, query):
        print(f"\nProcessing query: {query}")

        # Step 1: Determine if retrieval is necessary
        print("Step 1: Determining if retrieval is necessary...")
        input_data = {"query": query}
        retrieval_decision = 'yes'
        print(f"Retrieval decision: {retrieval_decision}")

        if retrieval_decision == 'yes':
            # Step 2: Retrieve relevant documents
            print("Step 2: Retrieving relevant documents...")
            docs = self.vectorstore.similarity_search(query, k=self.top_k)
            contexts = [doc.page_content for doc in docs]
            print(f"Retrieved {len(contexts)} documents")
            


            # Step 3: Evaluate relevance of retrieved documents
            print("Step 3: Evaluating relevance of retrieved documents...")
            relevant_contexts = []
            for i, context in enumerate(contexts):
                input_data = {"query": query, "context": context}
                relevance = self.relevance_chain.invoke(input_data).response.strip().lower()
                print(f"Document {i + 1} relevance: {relevance}")
                if relevance == 'relevant':
                    relevant_contexts.append(context)

            print(f"Number of relevant contexts: {len(relevant_contexts)}")

            # If no relevant contexts found, generate without retrieval
            if not relevant_contexts:
                print("No relevant contexts found. Generating without retrieval...")
                input_data = {"query": query, "context": "No relevant context found."}
                return self.generation_chain.invoke(input_data).response

            # Step 4: Generate response using relevant contexts
            print("Step 4: Generating responses using relevant contexts...")
            responses = []
            for i, context in enumerate(relevant_contexts):
                print(f"Generating response for context {i + 1}...")
                input_data = {"query": query, "context": context}
                response = self.generation_chain.invoke(input_data).response

                # Step 5: Assess support
                print(f"Step 5: Assessing support for response {i + 1}...")
                input_data = {"response": response, "context": context}
                support = self.support_chain.invoke(input_data).response.strip().lower()
                print(f"Support assessment: {support}")

                # Step 6: Evaluate utility
                print(f"Step 6: Evaluating utility for response {i + 1}...")
                input_data = {"query": query, "response": response}
                utility = int(self.utility_chain.invoke(input_data).response)
                print(f"Utility score: {utility}")

                responses.append((response, support, utility))

            # Select the best response based on support and utility
            print("Selecting the best response...")
            best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
            print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
            return best_response[0]
        else:
            # Generate without retrieval
            print("Generating without retrieval...")
            input_data = {"query": query, "context": "No retrieval necessary."}
            return self.generation_chain.invoke(input_data).response



