import json
import os
from typing import List, Tuple
import networkx as nx
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
import logging
from pathlib import Path
from src.kg_gen import Graph
import dspy

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# result = client.models.embed_content(
#         model="gemini-embedding-exp-03-07",
#         contents="What is the meaning of life?",
#         config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
# )
# print(result.embeddings)

# Set up logging safely
logger = logging.getLogger('kg_rag')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

class KGAssistedRAG:
    def __init__(self, knowledge_graph_path: str, openai_api_key: str = None):
        """
        Initialize KG-assisted RAG with cached embeddings, BM25 tokens, and text chunk store.
        """
        load_dotenv()
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        # self.kg = self._load_knowledge_graph(knowledge_graph_path)

        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Extract triples and texts
        # self.triples = [(u, d.get('relationship', 'related_to'), v) for u, v, d in self.kg.edges(data=True)]
        # self.triple_texts = [f"{h} {r} {t}" for h, r, t in self.triples]
        # print(f"Loaded {len(self.triples)} triples from knowledge graph")
        # print(f"Sample triple: {self.triples[0] if self.triples else 'No triples found'}")
        self.kg = kg_data
        self.nodes: list[str] = kg_data.get("entities")
        self.edges: list[str] = kg_data.get("edges")
        print(f"Loaded {len(self.nodes)} nodes from knowledge graph")
        print(f"Sample node: {self.nodes[0] if self.nodes else 'No nodes found'}")
        print(f"Loaded {len(self.edges)} edges from knowledge graph") 
        print(f"Sample edge: {self.edges[0] if self.edges else 'No edges found'}")
        
        # Cache embeddings and BM25 tokens
        self.node_embeddings = self.encoder.encode(self.nodes, show_progress_bar=True)
        self.node_bm25_tokenized = [text.lower().split() for text in self.nodes]
        self.node_bm25 = BM25Okapi(self.node_bm25_tokenized)
        
        self.edge_embeddings = self.encoder.encode(self.edges, show_progress_bar=True)
        self.edge_bm25_tokenized = [text.lower().split() for text in self.edges]
        self.edge_bm25 = BM25Okapi(self.edge_bm25_tokenized)
  
    def get_relevant_items(self, query: str, top_k: int = 50, type: str = "node") -> List[Tuple[str, str, str]]:
        """
        Use rank fusion of BM25 + embedding to retrieve top-k nodes.
        """
        query_tokens = query.lower().split()

        # BM25
        bm25_scores = self.node_bm25.get_scores(query_tokens) if type == "node" else self.edge_bm25.get_scores(query_tokens)
 
        # Embedding
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        embeddings = self.node_embeddings if type == "node" else self.edge_embeddings
        embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()

        # Rank fusion (equal weighting)
        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        top_items = [self.nodes[i] for i in top_indices]

        return top_items

    def deduplicate(self) -> Graph:
        kg = self.kg.copy()
        lm = dspy.LM(model="gemini/gemini-2.0-flash")
        dspy.configure(lm=lm)
        
        class DeduplicateEntities(dspy.Signature):
            """Deduplicate entities. Find and return duplicates and an alias that best represents the duplicates. Return an empty list if there are none. 
            Examples:
            TODO
            """
            entities: list[str] = dspy.InputField()
            duplicates: list[str] = dspy.OutputField()
            alias: list[str] = dspy.OutputField()
            
        deduplicate = dspy.Predict(DeduplicateEntities)
        deduplicated_entities = deduplicate(entities=self.nodes).duplicates
        
        class DeduplicateEdges(dspy.Signature):
            """Deduplicate edges. Find and return duplicates and an alias that best represents the duplicates. Return an empty list if there are none. 
            Examples:
            TODO
            """
            edges: list[str] = dspy.InputField()
            duplicates: list[str] = dspy.OutputField()
            alias: list[str] = dspy.OutputField()
            
        deduplicate_edges = dspy.Predict(DeduplicateEdges)
        deduplicated_edges = deduplicate_edges(edges=self.edges).duplicates
        
        
        graph = Graph(entities=deduplicated_entities, edges=deduplicated_edges)
        return graph
            
    
    
    # def generate_response(self, query: str, prompt: str = None, max_tokens: int = 500) -> str:
    #     """
    #     Prompt GPT with top relevant knowledge graph triples and their associated text evidence.
    #     """
    #     top_relations = self.get_relevant_nodes(query)
    #     triples_text = ""
    #     seen_chunks = set()
    #     ordered_chunks = []

    #     for s, r, t in top_relations:
    #         edge_data = self.kg.edges.get((s, t)) or self.kg.edges.get((t, s))  # undirected
    #         text_unit_ids = edge_data.get("text_unit_ids", []) if edge_data else []

    #         triples_text += f"({s}, {r}, {t}) | text_unit_ids: {text_unit_ids}\n"

    #         for tid in text_unit_ids:
    #             if tid in self.chunk_store and tid not in seen_chunks:
    #                 ordered_chunks.append(self.chunk_store[tid])
    #                 seen_chunks.add(tid)
    #                 if len(ordered_chunks) == 20:
    #                     break
    #         if len(ordered_chunks) == 20:
    #             break

    #     text_block = "\n\n".join(ordered_chunks)

    #     if prompt:
    #         final_prompt = (
    #             f"Here are knowledge graph triples and text evidence.\n\n"
    #             f"Triples:\n{triples_text}\n\n"
    #             f"Text Evidence:\n{text_block}\n\n{prompt}"
    #         )
    #     else:
    #         final_prompt = (
    #             f"Use the following knowledge graph triples and text evidence to answer the question.\n\n"
    #             f"Triples:\n{triples_text}\n\n"
    #             f"Text Evidence:\n{text_block}\n\n"
    #             f"Question: {query}\nAnswer:"
    #         )

    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant answering questions using the provided KG triples and associated text chunks."},
    #         {"role": "user", "content": final_prompt}
    #     ]

    #     response = self.client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=messages,
    #         max_tokens=max_tokens,
    #         temperature=0.2
    #     )

    #     return response.choices[0].message.content
    
if __name__ == "__main__":
    # # Initialize KGAssistedRAG with proper paths
    # rag = KGAssistedRAG(
    #     knowledge_graph_path="tests/data/wiki_qa/aggregated/articles_1_kg.json",
    #     openai_api_key=os.getenv('OPENAI_API_KEY')
    # )
    
    # # Test query
    # test_query = "Winter Olympics"
    
    # # Test get_relevant_nodes
    # nodes = rag.get_relevant_nodes(test_query)
    # print(f"\nQuery: {test_query}")
    # print(f"\nRelevant Nodes:")
    # for n in nodes:
    #     print(n)
    
    # Set paths
    kg_paths = [
        "tests/data/wiki_qa/aggregated/articles_1_kg.json",
        "tests/data/wiki_qa/aggregated/articles_40k_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_400k_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_4m_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_20m_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_w_context_kg.json",
    ]
    
    for kg_path in kg_paths:
        # Initialize KGAssistedRAG with proper paths
        rag = KGAssistedRAG(
            knowledge_graph_path=kg_path,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Test query
        test_query = "Winter Olympics"
        
        # Test get_relevant_nodes and edges
        nodes = rag.get_relevant_items(test_query, 50, "node")
        edges = rag.get_relevant_items(test_query, 50, "edge")
        print(f"\nQuery: {test_query}")
        print(f"\nRelevant Nodes:")
        for n in nodes:
            print(n)
        print(f"\nRelevant Edges:")
        for e in edges:
            print(e)
        
        graph = rag.deduplicate()
        
        # Construct deduplicated output path by adding _deduplicated before .json
        base, ext = os.path.splitext(kg_path)
        deduplicated_path = f"{base}_deduplicated{ext}"
        
        # Save deduplicated graph
        with open(deduplicated_path, "w") as f:
            json.dump(graph, f, indent=4)
            
        print(f"\nDeduplicated graph saved to {deduplicated_path}")
