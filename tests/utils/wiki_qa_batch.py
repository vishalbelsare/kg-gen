from concurrent.futures import ThreadPoolExecutor
import json
from typing import Literal
from src.kg_gen.kg_gen import KGGen, Graph
import pandas as pd
import os
import dspy

ARTICLES_DIR = "tests/data/wiki_qa/articles"
OUTPUT_KG_DIR = "tests/data/wiki_qa/articles_kg"


def generate_vanilla(kg, article, article_path):
    # Generate KG with chunking
    return kg.generate(
        input_data=article,
        chunk_size=2048,
        chunk_overlap=246,
        input_data_file_path=article_path
    )
    
def generate_w_context(kg, article, article_path, title):
    
    class SummarizeWikipediaArticle(dspy.Signature):
        "Explain what the text contents of the Wikipedia article cover"
        
        article_title: str = dspy.InputField()
        article_text: str = dspy.InputField()
        one_phrase_summary: str = dspy.OutputField()
        
    summarizer = dspy.Predict(SummarizeWikipediaArticle)

    summary = summarizer(article_title=title, article_text=article)
    print(f"Summary for article {title}: {summary.one_phrase_summary}")
    
    return kg.generate(
        input_data=article,
        chunk_size=2048,
        chunk_overlap=246,
        input_data_file_path=article_path,
        extraction_context=f'This is an excerpt from a Wikipedia article "{title}". {summary}'
    )

    
def generate_kgs_for_articles_with_chunks(thread_count: int = 1, articles_dir: str = ARTICLES_DIR, output_kg_dir: str = OUTPUT_KG_DIR):
    """Generate knowledge graphs for all articles in the articles directory."""
    kg = KGGen(
        model="gemini/gemini-2.0-flash-001",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.99
    )
    os.makedirs(output_kg_dir, exist_ok=True)

    # Get all article files
    article_files = [f for f in os.listdir(articles_dir) if f.endswith(".txt")]
    generation_errors = []

    def process_article(article_file, attempt: int = 0):
        article_path = os.path.join(articles_dir, article_file)
        title = os.path.splitext(article_file)[0]

        try:
            output_kg_path = os.path.join(output_kg_dir, f"{title}.json")
            if os.path.exists(output_kg_path):
                # print(f"KG already exists for '{title}'")
                return {"status": "skipped", "title": title}

            with open(article_path, "r") as f:
                article = f.read()

            # 💛 sub this out for experiments 💛
            graph_chunked = generate_w_context(kg, article, article_path, title)

            with open(output_kg_path, "w") as f:
                f.write(graph_chunked.model_dump_json(indent=4))
            print(f"Saved knowledge graph for '{title}' to {output_kg_path}")
            return {"status": "success", "title": title}
        except Exception as e:
            # if attempt < 20:
            #     print(f"Retrying {article_file} ({attempt + 1}/10)")
            #     return process_article(article_file, attempt + 1)
            error_info = {"title": title, "error": str(e)}
            print(f"Error generating KG for '{title}': {e}")
            return {"status": "generation_error", "data": error_info}

    # Process articles based on thread count
    if thread_count <= 1:
        # Process articles sequentially without threading
        results = [process_article(article_file) for article_file in article_files]
    else:
        # Process articles using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(executor.map(process_article, article_files))

    # Process results
    for result in results:
        if result["status"] == "generation_error":
            generation_errors.append(result["data"])

    # Save error logs
    if generation_errors:
        errors_path = os.path.join(output_kg_dir, "_all_articles_generation_errors.json")
        with open(errors_path, "w") as f:
            json.dump(generation_errors, f, indent=4)
        print(
            f"Saved {len(generation_errors)} generation error records to {errors_path}"
        )

    print(
        f"Processed {len(article_files)} articles, with {len(generation_errors)} errors"
    )


def aggregate_all_kgs(output_kg_dir: str):
    """
    Aggregate all knowledge graphs in the specified output_kg_dir into a single combined graph.

    Args:
        output_kg_dir (str): The directory containing the knowledge graph files.

    Returns:
        Graph: The combined knowledge graph, or None if no valid graphs are found.
    """

    print(f"Aggregating all knowledge graphs from {output_kg_dir}...")

    # Initialize KGGen
    kg = KGGen()

    # Get all KG files
    kg_files = []
    for root, _, files in os.walk(output_kg_dir):
        for file in files:
            if file.endswith(".json") and not file.endswith("_errors.json") and not file.endswith('combined_graph.json'):
                kg_files.append(os.path.join(root, file))

    print(f"Found {len(kg_files)} knowledge graph files")

    # Load all graphs
    graphs = []
    for kg_file in kg_files:
        try:
            with open(kg_file, "r") as f:
                graph_data = json.load(f)

            # Create Graph object from the loaded data
            graph = Graph(
                entities=set(graph_data.get("entities", [])),
                relations=set(tuple(r) for r in graph_data.get("relations", [])),
                edges=set(graph_data.get("edges", [])),
                entities_chunk_ids=graph_data.get("entities_chunk_ids", {}),
                relations_chunk_ids=graph_data.get("relations_chunk_ids", {}),
                edges_chunk_ids=graph_data.get("edges_chunk_ids", {}),
            )
            graphs.append(graph)
        except Exception as e:
            print(f"Error loading graph from {kg_file}: {e}")

    # Aggregate all graphs
    if graphs:
        combined_graph = kg.aggregate(graphs)

        # Save the combined graph
        output_path = os.path.join(output_kg_dir, "_combined_graph.json")
        with open(output_path, "w") as f:
            f.write(combined_graph.model_dump_json(indent=4))

        print(f"Combined graph saved to {output_path}")
        print(
            f"Combined graph stats: {len(combined_graph.entities)} entities, {len(combined_graph.relations)} relations"
        )

        return combined_graph
    else:
        print("No valid graphs found to aggregate")
        return None



if __name__ == "__main__":
    # Generate KGs for all article directories
    article_dirs = [
        # "tests/data/wiki_qa/articles_1"
        # "tests/data/wiki_qa/articles_40k_ch",
        # "tests/data/wiki_qa/articles_400k_ch",
        # "tests/data/wiki_qa/articles_4m_ch",
        # "tests/data/wiki_qa/articles_20m_ch",
        # "tests/data/wiki_qa/articles"
        "tests/data/wiki_qa/articles_w_context"
    ]
    
    for article_dir in article_dirs:
        kg_dir = article_dir + "_kg"
        print(f"\nProcessing articles from {article_dir}")
        generate_kgs_for_articles_with_chunks(thread_count=128, articles_dir=article_dir, output_kg_dir=kg_dir)
    
        aggregate_all_kgs(output_kg_dir=kg_dir)
