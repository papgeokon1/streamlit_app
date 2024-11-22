import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import AIMessage
import diskcache
from functools import lru_cache
import cProfile
from keybert import KeyBERT
import pstats
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import os
import sys

from langchain_openai import ChatOpenAI
from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
import heapq
from langchain.schema import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor ,as_completed
from tqdm import tqdm
import numpy as np
import aiohttp
from spacy.cli import download
from spacy.lang.en import English
from functools import lru_cache
import pickle
import hashlib
import joblib
import io
from transformers import pipeline, logging
from sentence_transformers import SentenceTransformer
from pathlib import Path
import streamlit as st

# Από τα secrets του Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]


# Αρχικοποίηση μοντέλου περίληψης
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


from helper_functions import *
from evaluate_rag import *
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

import re


def clean_content(text: str) -> str:
    cleaned_text = re.sub(r"additional_kwargs=.*?}}", "", text, flags=re.DOTALL)
    cleaned_text = re.sub(r"response_metadata=.*?}}", "", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"token_usage=.*?}}", "", cleaned_text, flags=re.DOTALL)

    cleaned_text = re.sub(r"[{}]", "", cleaned_text)
    return cleaned_text.strip()

import aiofiles
from langchain_community.document_loaders import PDFMinerLoader
from pdfminer.high_level import extract_text
from io import BytesIO



class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()

    def process_documents(self, documents):
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store


    def create_embeddings_parallel(self, texts, batch_size=32):
        """
        Δημιουργία embeddings με παράλληλη επεξεργασία.
        """
        def process_batch(batch):
            return self.embeddings.embed_documents(batch)

        # Χωρισμός του κειμένου σε δέσμες
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        embeddings = []

        # Εκτέλεση παράλληλα με ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_batch, batches))

        # Συνένωση των αποτελεσμάτων
        for result in results:
            embeddings.extend(result)

        return np.array(embeddings)

    def compute_similarity_matrix_parallel(self, embeddings, batch_size=100):
        """
        Υπολογισμός similarity matrix με batch και παράλληλη επεξεργασία.
        """
        num_embeddings = len(embeddings)
        similarity_matrix = np.zeros((num_embeddings, num_embeddings))

        def process_batch(batch_indices):
            local_matrix = np.zeros((len(batch_indices), num_embeddings))
            for i, idx in enumerate(batch_indices):
                local_matrix[i] = cosine_similarity([embeddings[idx]], embeddings)[0]
            return batch_indices, local_matrix

        # Υπολογισμός σε batches
        with ProcessPoolExecutor() as executor:
            batch_indices_list = [range(i, min(i + batch_size, num_embeddings)) for i in range(0, num_embeddings, batch_size)]
            results = list(executor.map(process_batch, batch_indices_list))

        # Ενοποίηση αποτελεσμάτων
        for batch_indices, local_matrix in results:
            for i, idx in enumerate(batch_indices):
                similarity_matrix[idx] = local_matrix[i]

        return similarity_matrix

class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")

import shutil
import tempfile

class KnowledgeGraph:
    def __init__(self, graph_filepath='graph.pkl', embeddings_filepath='embeddings.pkl', concept_cache_filepath='concept_cache.pkl'):
        self.graph_filepath = graph_filepath
        self.embeddings_filepath = embeddings_filepath
        self.concept_cache_filepath = concept_cache_filepath

        # Load or initialize data
        self.graph = self._load_from_file(self.graph_filepath, default=nx.Graph())
        self.embeddings = self._load_from_file(self.embeddings_filepath, default=None)
        self.concept_cache = self._load_from_file(self.concept_cache_filepath, default={})
        lightweight_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.kw_model = KeyBERT()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8
        self.node_content_hashes = set()

    def clear_cache(self):
        """Clears the graph, embeddings, and concept cache files."""
        for filepath in [self.graph_filepath, self.embeddings_filepath, self.concept_cache_filepath]:
            if os.path.exists(filepath):
                os.remove(filepath)
        # Reinitialize the data
        self.graph = nx.Graph()
        self.embeddings = None
        self.concept_cache = {}
        print("Cache cleared and graph reinitialized.")

    def _save_to_file(self, obj, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)

    def _load_from_file(self, filepath, default):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        return default

    def build_graph(self, splits, llm, embedding_model, force_rebuild=True):
        if not self.graph.nodes or force_rebuild:
            if force_rebuild:
                self.clear_cache()  # Clear cache if forced to rebuild

            self._add_nodes(splits)
            embeddings = self._create_embeddings(splits, embedding_model)
            self.embeddings = embeddings  # Store the embeddings for later retrieval
            self._extract_concepts(splits, llm)
            self._add_edges(embeddings)

            # Save the updated graph, embeddings, and concept cache
            self._save_to_file(self.graph, self.graph_filepath)
            self._save_to_file(self.embeddings, self.embeddings_filepath)
            self._save_to_file(self.concept_cache, self.concept_cache_filepath)
        else:
            print("Graph, embeddings, and concepts loaded from cache.")

    def _extract_keyword(self, text, top_n=1):
        """
        Εξαγωγή λέξεων-κλειδιών χρησιμοποιώντας το KeyBERT.
        """
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),  # Μόνο μεμονωμένες λέξεις
            stop_words='english',
            top_n=top_n  # Επιλογή έως και N keywords
        )
        return keywords[0][0] if keywords else "No Keyword"

    def _add_nodes(self, splits):
        node_count = {}
        for i, split in enumerate(splits):
            content_hash = hashlib.md5(split.page_content.encode('utf-8')).hexdigest()

            if content_hash not in self.node_content_hashes:
                # Εξαγωγή του keyword ως label
                base_name = self._extract_keyword(split.page_content)
                if base_name in node_count:
                    node_count[base_name] += 1
                    unique_node_name = f"{base_name} ({node_count[base_name]})"
                else:
                    node_count[base_name] = 1
                    unique_node_name = base_name

                self.graph.add_node(i, content=split.page_content, label=unique_node_name)
                self.node_content_hashes.add(content_hash)
            else:
                print(f"Duplicate node detected and skipped: {split.page_content[:100]}...")



    def _create_embeddings(self, splits, embedding_model):
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)
    
    def _compute_similarities(self, embeddings):
        return cosine_similarity(embeddings)

    def _extract_concepts(self, splits, llm, max_threads=8):
        """
        Παράλληλη εξαγωγή concepts με περιορισμένο αριθμό threads και caching.
        """
        with ThreadPoolExecutor(max_threads) as executor:
            # Δημιουργία εργασιών
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                            for i, split in enumerate(splits)}

            # Επεξεργασία αποτελεσμάτων
            for future in tqdm(as_completed(future_to_node), total=len(splits),
                            desc="Extracting concepts and entities"):
                node = future_to_node[future]
                try:
                    concepts = future.result()
                    self.graph.nodes[node]['concepts'] = concepts
                except Exception as e:
                    print(f"Error in concept extraction for node {node}: {e}")


    def _load_spacy_model(self):
        try:
            print(os.path.exists("./models/en_core_web_sm"))
            return spacy.load("./models/en_core_web_sm")
        except OSError:
            # Αν δεν υπάρχει, κατέβασε το και φόρτωσε το
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")


    def _extract_concepts_and_entities(self, content, llm):
        if content in self.concept_cache:
            return self.concept_cache[content]

        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        all_concepts = list(set(named_entities + general_concepts))
        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm, max_threads=8):
        """
        Παράλληλη εξαγωγή concepts με περιορισμένο αριθμό threads και caching.
        """
        with ThreadPoolExecutor(max_threads) as executor:
            # Δημιουργία εργασιών
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                            for i, split in enumerate(splits)}

            # Επεξεργασία αποτελεσμάτων
            for future in tqdm(as_completed(future_to_node), total=len(splits),
                            desc="Extracting concepts and entities"):
                node = future_to_node[future]
                try:
                    concepts = future.result()
                    self.graph.nodes[node]['concepts'] = concepts
                except Exception as e:
                    print(f"Error in concept extraction for node {node}: {e}")

    def _add_edges(self, embeddings):
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])
    
class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer to the query")
    answer: str = Field(description="The current answer based on the context, if any")

class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs, max_depth=3, min_similarity_threshold=0.75) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        Expands the context by traversing the knowledge graph using a Dijkstra-like approach.
        Now includes depth and similarity threshold limits for optimization.
        """
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        
        if len(self.knowledge_graph.graph.nodes) == 0:
            print("Knowledge graph is empty. Cannot perform expansion.")
            return expanded_context, traversal_path, filtered_content, "Graph is empty, unable to provide an answer."
        
        priority_queue = []
        g_score = {}
        f_score = {}
        came_from = {}

        print("\nTraversing the knowledge graph:")

        for doc in relevant_docs:
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

       
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)
            if closest_node is None:
                print("No closest node found. Skipping this document.")
                continue

            # Initial cost to reach closest_node is 0
            g_score[closest_node] = 0
            # Heuristic h(n) based on semantic similarity (lower similarity implies higher heuristic cost)
            f_score[closest_node] = self._heuristic(closest_node, query)

            heapq.heappush(priority_queue, (f_score[closest_node], closest_node))

        step = 0
        while priority_queue and step < max_depth:
            current_f, current_node = heapq.heappop(priority_queue)

            if current_node in visited_concepts:
                continue

            visited_concepts.add(current_node)
            traversal_path.append(current_node)

            node_content = self.knowledge_graph.graph.nodes[current_node]['content']
            filtered_content[current_node] = node_content
            expanded_context += "\n" + node_content if expanded_context else node_content

            # Check if the context contains a complete answer
            is_complete, answer = self._check_answer(query, expanded_context)
            if is_complete:
                final_answer = answer
                break

            # Explore neighbors
            for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                edge_weight = self.knowledge_graph.graph[current_node][neighbor]['weight']
                tentative_g_score = g_score.get(current_node, float('inf')) + edge_weight

                # Check if we have found a better path to the neighbor
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, query)

                    heapq.heappush(priority_queue, (f_score[neighbor], neighbor))

            step += 1

        # If no complete answer was found, generate a final response based on the expanded context
        if not final_answer:
            final_answer = self._generate_final_answer(query, expanded_context)

        return expanded_context, traversal_path, filtered_content, final_answer

    def _heuristic(self, node, query):
        """
        Heuristic function for A* search, measuring semantic similarity between the node and the query.
        Lower values indicate a closer match.
        """
        node_concepts = set(self.knowledge_graph.graph.nodes[node]['concepts'])
        query_concepts = self._extract_query_concepts(query)
        
        # Heuristic based on concept overlap (higher overlap -> lower heuristic cost)
        intersection = node_concepts.intersection(query_concepts)
        union = node_concepts.union(query_concepts)

        # Return a normalized similarity score (lower score is better for A*)
        return 1 - (len(intersection) / len(union)) if union else 1
    
    def _extract_query_concepts(self, query):
        """
        Extract key concepts from the query using the same method as node concept extraction.
        """
        # Extract concepts from the query (this can be refined depending on your setup)
        return set(query.lower().split())  # Simple token-based splitting for demonstration

    def _generate_final_answer(self, query, expanded_context):
        """
        Generate the final answer using the expanded context and the LLM.
        """
        response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
        )
        response_chain = response_prompt | self.llm
        input_data = {"query": query, "context": expanded_context}
        final_answer = response_chain.invoke(input_data)

        # Ensure the final answer is clean and properly formatted
        if isinstance(final_answer, AIMessage):
            final_answer = final_answer.content
        final_answer = clean_content(final_answer)

        return final_answer
    
    
    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        Handles the query by retrieving relevant information from the knowledge graph.
        
        Args:
        - query (str): The query to be answered.
        
        Returns:
        - str: The final answer as a response.
        """
        print(f"\nProcessing query: {query}")
        relevant_docs = self._retrieve_relevant_documents(query)
        expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)

        # Clean the final answer before returning
        final_answer = clean_content(final_answer)
        return final_answer, traversal_path, filtered_content
   
   
    def _retrieve_relevant_documents(self, query: str):
        print("\nRetrieving relevant documents...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        return compression_retriever.invoke(query)


# The rest of the GraphRAG implementation goes here, following the structure of your original file.
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class Visualizer:
    @staticmethod
    def display_node_content(graph, node_id):
        """
        Εμφανίζει το περιεχόμενο ενός κόμβου, μετατρέποντάς το σε συντακτικά ορθό και συνοπτικό κείμενο με το BART.

        Args:
            graph (nx.Graph): Ο γράφος.
            node_id (int): Το ID του κόμβου.

        Returns:
            str: Το περιληπτικό κείμενο.
        """
        # Λήψη του περιεχομένου του κόμβου
        content = graph.nodes[node_id].get('content', '')

        # Μετατροπή σε plain text και περίληψη
        if isinstance(content, str) and len(content) > 100:  # Εφαρμόζεται μόνο αν το κείμενο είναι αρκετά μεγάλο
            try:
                summarized_text = summarizer(content[:1000], max_length=50, min_length=15, do_sample=False)
                plain_text = summarized_text[0]['summary_text']
            except Exception as e:
                print(f"Error in summarization: {e}")
                plain_text = content[:150]  # Σε περίπτωση σφάλματος, χρησιμοποιείται ένα απόσπασμα
        else:
            plain_text = content

        return plain_text


    @staticmethod

    def create_subgraph(self):
        # Extract a subgraph that contains only the nodes in the traversal path
        subgraph = self.graph_rag.graph.subgraph(self.graph_rag.traversal_path)
        return subgraph

        
    def visualize_traversal(graph, traversal_path):
        """
        Visualizes the traversal path on the knowledge graph with nodes, edges, and traversal path highlighted.

        Args:
        - graph (networkx.Graph): The knowledge graph containing nodes and edges.
        - traversal_path (list of int): The list of node indices representing the traversal path.

        Returns:
        - None
        """
        traversal_graph = nx.DiGraph()

        # Add nodes and edges from the original graph
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate positions for all nodes
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # Draw regular edges with color based on weight
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)

        # Draw traversal path with curved arrows
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # Calculate control point for curve
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # Draw curved arrow
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',
                                            linewidth=2,
                                            zorder=4)
            ax.add_patch(arrow)

        # Prepare labels for the nodes
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # Draw labels
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        # Highlight start and end nodes
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[start_node],
                               node_color='lightgreen',
                               node_size=3000,
                               ax=ax)

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[end_node],
                               node_color='lightcoral',
                               node_size=3000,
                               ax=ax)

        ax.set_title("Graph Traversal Flow")
        ax.axis('off')

        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                   norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)

        # Add legend
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Regular Edge')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15,
                                 label='Start Node')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15,
                               label='End Node')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left',
                            bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    def visualize_subgraph(self, subgraph):
        # Δημιουργία layout για το υπογράφημα
        pos = nx.spring_layout(subgraph)

        # Εξαγωγή labels κόμβων (χρήση του 'label' αντί για 'concepts' ή 'content')
        labels = {node: data.get('label', 'Unknown') for node, data in subgraph.nodes(data=True)}
        
        # Καθορισμός του πρώτου και του τελευταίου κόμβου
        start_node = list(subgraph.nodes())[0]
        end_node = list(subgraph.nodes())[-1]
        
        # Σχεδίαση του υπογραφήματος με τα labels
        nx.draw(subgraph, pos, with_labels=True, labels=labels, node_color='lightblue', font_weight='bold')
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[start_node], node_color='green', node_size=3000, label="Start")
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[end_node], node_color='red', node_size=3000, label="End")
       
        # Διαχείριση και εμφάνιση βαρών ακμών
        edge_labels = nx.get_edge_attributes(subgraph, 'weight')  # Απόκτηση βαρών ακμών
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)


        # Εμφάνιση του γράφου
        plt.legend(["Start Node", "End Node"])        
        st.pyplot(plt)

    
    
    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        Prints the filtered content of visited nodes in the order of traversal.
    
     Args:
        - traversal_path (list of int): The list of node indices representing the traversal path.
        - filtered_content (dict of int: str): A dictionary mapping node indices to their filtered content.
    
     Returns:
        - None
        """
        print("\nFiltered content of visited nodes in order of traversal:")
        for i, node in enumerate(traversal_path):
            print(f"\nStep {i + 1} - Node {node}:")
        # Print the node identifier (or part of the node content) instead of concepts
            print(f"Content: {filtered_content.get(node, 'No content available')[:200]}...")  # Print first 200 characters
            print("-" * 50)


# Define the graph RAG class
class GraphRAG:
    def __init__(self, urls, pdf_files, json_files=None,jsonl_files=None , html_files=None, csv_files=None,txt_files=None,direct_txt_content=""):
        self.urls = urls
        self.pdf_files = pdf_files
        self.json_files = json_files
        self.jsonl_files= jsonl_files
        self.html_files = html_files
        self.csv_files = csv_files
        self.txt_files=txt_files
        self.direct_txt_content=direct_txt_content
        self.vectorstore = None
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        self.embedding_model = OpenAIEmbeddings()
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None
        self.visualizer = Visualizer()  
        self.graph=nx.Graph()
        self.traversal_path=[] # New: Stores the nodes that are part of the answer        

    async def initialize(self):
        combined_content = ""
        tasks = []  # Λίστα για τις ασύγχρονες εργασίες

        # Προσθήκη εργασιών για URLs
        if self.urls:
            for url in self.urls:
                print(f"Fetching content from: {url}")
                tasks.append(fetch_text_from_url(url))

        # Προσθήκη εργασιών για JSON αρχεία
        if self.json_files:
            for json_path in self.json_files:
                print(f"Processing JSON file: {json_path}")
                tasks.append(fetch_text_from_json(json_path))

        # Προσθήκη εργασιών για JSONL αρχεία
        if self.jsonl_files:
            for jsonl_path in self.jsonl_files:
                print(f"Processing JSONL file: {jsonl_path}")
                tasks.append(fetch_text_from_jsonl(jsonl_path))

        # Προσθήκη εργασιών για HTML αρχεία
        if self.html_files:
            for html_path in self.html_files:
                print(f"Processing HTML file: {html_path}")
                tasks.append(fetch_text_from_html(html_path))

        # Προσθήκη εργασιών για CSV αρχεία
        if self.csv_files:
            for csv_path in self.csv_files:
                print(f"Processing CSV file: {csv_path}")
                tasks.append(fetch_text_from_csv(csv_path))

        # Προσθήκη εργασιών για TXT αρχεία
        if self.txt_files:
            for txt_path in self.txt_files:
                print(f"Processing TXT file: {txt_path}")
                tasks.append(fetch_text_from_txt(txt_path))

        # Εκτέλεση όλων των εργασιών παράλληλα και συνένωση των αποτελεσμάτων
        contents = await self._fetch_all_contents(tasks)

        # Προσθήκη του περιεχομένου των αρχείων στις συνενώσεις
        for content in contents:
            if content:
                combined_content += content + "\n\n"

        # Συγχρονισμένη επεξεργασία PDF αρχείων με PDFMinerLoader
        if self.pdf_files:
            for pdf_path in self.pdf_files:
                print(f"Processing PDF file: {pdf_path}")
                pdf_loader = PDFMinerLoader(pdf_path)
                pdf_docs = pdf_loader.load()

                for pdf_doc in pdf_docs:
                    combined_content += pdf_doc.page_content + "\n\n"

        # Προσθήκη άμεσου κειμένου από το text area αν υπάρχει
        if self.direct_txt_content:
            combined_content += self.direct_txt_content + "\n\n"

        # Δημιουργία του vectorstore αν υπάρχει περιεχόμενο
        if combined_content:
            self.vectorstore = encode_from_string(combined_content)
            # Διαδικασία επεξεργασίας εγγράφων και κατασκευή knowledge graph
            self.process_documents(combined_content)
        else:
            raise ValueError("Failed to retrieve or process the content from URLs or PDFs.")

    async def _fetch_all_contents(self, tasks):
        """
        Εκτελεί όλες τις εργασίες ανάκτησης περιεχομένου παράλληλα και επιστρέφει τα αποτελέσματα.
        """
        return await asyncio.gather(*tasks)
        
    def process_documents(self, content: str):
        """
        Processes the combined content by splitting it into chunks, embedding them, and building a knowledge graph.
        """
        documents = [Document(page_content=content)]
        splits, self.vectorstore = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(self.vectorstore, self.knowledge_graph, self.llm)        
    
    def retrieve_relevant_nodes(self, query):
        # Get the embedding of the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Ensure embeddings exist in the knowledge graph
        if self.knowledge_graph.embeddings is None:
            print("Embeddings are not available in the knowledge graph.")
            return []
        
        # Compute similarity scores between the query embedding and document embeddings
        similarity_scores = cosine_similarity([query_embedding], self.knowledge_graph.embeddings)[0]
        
        # Find the top-k most relevant nodes (documents)
        top_k_indices = similarity_scores.argsort()[-5:][::-1]  # Get top 5 nodes with the highest similarity scores
        
        # Return the node indices as the relevant nodes
        relevant_nodes = top_k_indices.tolist()
        return relevant_nodes

    def query(self, query: str):
        """
        Handles a query by retrieving relevant information from the knowledge graph.
        
        Args:
        - query (str): The query to be answered.
        
        Returns:
        - str: The final answer as a response.
        """
        # Debugging print για έλεγχο του vectorstore και του graph
        print("Vectorstore status:", "Initialized" if self.vectorstore else "Not Initialized")
        print("Knowledge graph nodes:", len(self.knowledge_graph.graph.nodes))
        
        # Reset the traversal path for every new query
        self.traversal_path = []        
        
        # Process the query και ανάκτηση σχετικών κόμβων
        relevant_nodes = self.retrieve_relevant_nodes(query)
        
        if not relevant_nodes:
            print("No relevant nodes found in the graph.")
        
        # Track the traversal path
        self.traversal_path = relevant_nodes
        
        # Ελέγχει αν υπάρχει query_engine και προχωρά με την ερώτηση
        if self.query_engine is None:
            print("Query Engine is not initialized.")
            return None, None
        
        final_answer, traversal_path, filtered_content = self.query_engine.query(query)
        
        # Visualize if there is traversal path
        if self.traversal_path:
            subgraph = self.knowledge_graph.graph.subgraph(self.traversal_path)
        else:
            subgraph = None
        
        final_answer = clean_content(final_answer)
        
        return final_answer, subgraph