"""
Obsidian Notes Knowledge Graph Generator

This script analyzes markdown files in Obsidian note style and generates
a knowledge graph representing the relationships between facts, summaries,
keyphrases, and notes.
"""
import os
import glob
from typing import List, Dict, Tuple, Set
from src.kg_gen import KGGen

# Define entity and edge types for Obsidian notes
ENTITY_TYPES = [
  "Fact",        # A discrete piece of information
  "Summary",     # A condensed explanation of some content
  "KeyPhrase",   # An important term or concept
  "Note"         # A full note or session
]

# Define suggested edge types (not strictly required)
EDGE_TYPES = [
  "is_predicate",    # KeyPhrase → KeyPhrase relationship
  "is_summary_of",   # Summary → Fact relationship
  "is_session_of",   # Note → Fact/Summary/KeyPhrase relationship
  "occurs_in",       # KeyPhrase → Fact/Summary relationship
  "is_similar"       # KeyPhrase → KeyPhrase, Fact → Fact, Summary → Summary similarity
]

# Note: While entities must have one of the specified types,
# edges can have one of the suggested types or no specific type at all.

def read_markdown_files(directory_path: str) -> Dict[str, str]:
    """
    Read all markdown files from the specified directory
    
    Args:
        directory_path: Path to directory containing markdown files
        
    Returns:
        Dictionary mapping filename (without extension) to file content
    """
    files_dict = {}
    markdown_files = glob.glob(os.path.join(directory_path, "*.md"))
    
    for file_path in markdown_files:
        filename = os.path.basename(file_path)
        note_name = os.path.splitext(filename)[0]
        
        with open(file_path, 'r') as f:
            content = f.read()
            files_dict[note_name] = content
            
    return files_dict

def process_single_file(file_path: str):
    """
    Process a single markdown file to generate and save a knowledge graph
    
    Args:
        file_path: Path to the markdown file
    """
    # Initialize KGGen
    kg_gen = KGGen(model="openai/gpt-4o", temperature=0.0)
    
    # Get note name from file
    filename = os.path.basename(file_path)
    note_name = os.path.splitext(filename)[0]
    print(f"Processing {note_name}...")
    
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "output", note_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a custom implementation that avoids the Literal type issue
    try:
        # Extract entities and relations directly with DSPy to avoid the Literal type error
        import dspy
        
        # Initialize DSPy with the model
        dspy_module = kg_gen.dspy
        
        class CustomEntitiesWithTypes(dspy.Signature):
            """Extract key entities from the Obsidian-style markdown note.
            This is for building a knowledge graph from notes. 
            Each entity must have one of the allowed entity types."""
            
            text: str = dspy.InputField()
            allowed_types: list[str] = dspy.InputField(desc="List of allowed entity types")
            entities: list[dict] = dspy.OutputField(desc="List of entities with their types as {entity: str, type: str}")
        
        class CustomRelationsWithTypes(dspy.Signature):
            """Extract relationships between entities in the Obsidian-style markdown note.
            This is for building a knowledge graph from notes.
            Relationships can be of the allowed types, but can also have no specific type."""
            
            text: str = dspy.InputField()
            entities: list[dict] = dspy.InputField(desc="List of entities with their types")
            allowed_relation_types: list[str] = dspy.InputField(desc="List of suggested relationship types (optional)")
            relations: list[dict] = dspy.OutputField(desc="List of relations as {subject: str, predicate: str, object: str, type: str (optional)}")
        
        # Extract entities with types (with a timeout to handle large files)
        extract_entities = dspy.Predict(CustomEntitiesWithTypes)
        try:
            entities_result = extract_entities(
                text=content[:20000],  # Limit content size to avoid timeouts
                allowed_types=ENTITY_TYPES
            )
        except Exception as e:
            print(f"Error extracting entities, trying with shorter content: {str(e)}")
            # Try with even shorter content if the first attempt fails
            entities_result = extract_entities(
                text=content[:10000], 
                allowed_types=ENTITY_TYPES
            )
        
        # Prepare entities for relation extraction
        entities = [item["entity"] for item in entities_result.entities]
        entity_types = {item["entity"]: item["type"] for item in entities_result.entities if "type" in item}
        
        # Skip if no entities found
        if not entities or len(entities) == 0:
            print(f"No entities found in {note_name}, skipping...")
            return None
        
        # Extract relations with types
        extract_relations = dspy.Predict(CustomRelationsWithTypes)
        relations_result = extract_relations(
            text=content[:20000],
            entities=entities_result.entities,
            allowed_relation_types=EDGE_TYPES
        )
        
        # Prepare relations
        relations = []
        relation_types = {}
        
        for rel in relations_result.relations:
            if "subject" in rel and "predicate" in rel and "object" in rel:
                relations.append((rel["subject"], rel["predicate"], rel["object"]))
                if "type" in rel:
                    relation_types[rel["predicate"]] = rel["type"]
        
        # Create a graph manually
        from src.kg_gen.models import Graph
        
        graph = Graph(
            entities=set(entities),
            relations=set(relations),
            edges={rel[1] for rel in relations},
            entity_types=entity_types,
            edge_types=relation_types
        )
        
        # Save the graph
        import json
        
        graph_dict = {
            'entities': list(graph.entities),
            'relations': [list(r) for r in graph.relations],
            'edges': list(graph.edges),
            'entity_types': graph.entity_types,
            'edge_types': graph.edge_types
        }
        
        with open(os.path.join(output_dir, 'graph.json'), 'w') as f:
            json.dump(graph_dict, f, indent=2)
            
        return graph
        
    except Exception as e:
        print(f"Error in custom implementation for {note_name}: {str(e)}")
        
        # Fallback to a simpler approach that doesn't use the Pydantic Literal type
        try:
            # Use a simpler approach without Pydantic models that use Literal
            content_with_context = f"""
Obsidian Notes Knowledge Graph
-----------------------------
Entity Types: {', '.join(ENTITY_TYPES)}
Relationship Types: {', '.join(EDGE_TYPES)}
-----------------------------

{content}
"""
            # Generate a simple graph without clustering to avoid the error
            note_graph = kg_gen.generate(
                input_data=content_with_context,
                context="Obsidian-style notes knowledge graph. Extract entities and relationships from the note. Edges can be one of the specified types but don't require a type.",
                node_type=ENTITY_TYPES,
                edge_type=EDGE_TYPES,
                require_node_type=True,
                require_edge_type=False,
                cluster=False,  # Skip clustering to avoid additional errors
                output_folder=output_dir
            )
            
            return note_graph
            
        except Exception as e2:
            print(f"Both approaches failed for {note_name}: {str(e2)}")
            return None

def generate_obsidian_notes_kg():
    """
    Generate knowledge graph from Obsidian-style markdown notes
    """
    # Get all markdown files
    notes_dir = os.path.join(os.path.dirname(__file__), "notes")
    markdown_files = glob.glob(os.path.join(notes_dir, "*.md"))
    
    # Process all the markdown files in the notes directory
    if markdown_files:
        print(f"Found {len(markdown_files)} markdown files, processing all files")
    else:
        print("No markdown files found in directory:", notes_dir)
    
    # Process each file individually with progress tracking
    all_graphs = []
    total_files = len(markdown_files)
    successful = 0
    failed = 0
    
    print(f"Starting to process {total_files} markdown files...")
    
    for i, file_path in enumerate(markdown_files):
        try:
            print(f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
            graph = process_single_file(file_path)
            if graph:
                all_graphs.append(graph)
                successful += 1
                print(f"Successfully processed {os.path.basename(file_path)} ({successful} successful, {failed} failed, {i+1}/{total_files} completed)")
            else:
                failed += 1
                print(f"No graph generated for {os.path.basename(file_path)} ({successful} successful, {failed} failed, {i+1}/{total_files} completed)")
        except Exception as e:
            failed += 1
            print(f"Error processing file {file_path}: {str(e)}")
            print(f"Progress: {successful} successful, {failed} failed, {i+1}/{total_files} completed")
    
    # Only aggregate if we have graphs
    if all_graphs:
        # Initialize KGGen for aggregation
        kg_gen = KGGen(model="openai/gpt-4o", temperature=0.0)
        
        # Aggregate all individual graphs into a combined notes graph
        notes_graph = kg_gen.aggregate(all_graphs)
        
        # Save the aggregated graph
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a comprehensive representation of the graph for saving
        import json
        
        # Build a more detailed graph representation with typing information
        graph_dict = {
            'entities': list(notes_graph.entities),
            'relations': [list(r) for r in notes_graph.relations],
            'edges': list(notes_graph.edges),
        }
        
        # Include entity type information if available
        if hasattr(notes_graph, 'entity_types') and notes_graph.entity_types:
            graph_dict['entity_types'] = notes_graph.entity_types
        
        # Include edge type information if available
        if hasattr(notes_graph, 'edge_types') and notes_graph.edge_types:
            graph_dict['edge_types'] = notes_graph.edge_types
            
        # Include clustering information if available
        if hasattr(notes_graph, 'entity_clusters') and notes_graph.entity_clusters:
            graph_dict['entity_clusters'] = {rep: list(cluster) for rep, cluster in notes_graph.entity_clusters.items()}
        if hasattr(notes_graph, 'edge_clusters') and notes_graph.edge_clusters:
            graph_dict['edge_clusters'] = {rep: list(cluster) for rep, cluster in notes_graph.edge_clusters.items()}
        
        with open(os.path.join(output_dir, 'notes_graph.json'), 'w') as f:
            json.dump(graph_dict, f, indent=2)
        
        print("\n===== KNOWLEDGE GRAPH GENERATION COMPLETE =====")
        print(f"Obsidian notes knowledge graph generated in {output_dir}")
        print(f"Files processed: {total_files} total, {successful} successful, {failed} failed")
        print(f"Knowledge Graph Statistics:")
        print(f"  - Total entities: {len(notes_graph.entities)}")
        print(f"  - Total relations: {len(notes_graph.relations)}")
        print(f"  - Total edge types: {len(notes_graph.edges)}")
        
        # Print entity type distribution if available
        if hasattr(notes_graph, 'entity_types') and notes_graph.entity_types:
            type_counts = {}
            for entity_type in notes_graph.entity_types.values():
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            print("\nEntity type distribution:")
            for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {entity_type}: {count} entities")
        
        # Print relation type distribution if available
        if hasattr(notes_graph, 'edge_types') and notes_graph.edge_types:
            relation_counts = {}
            for rel_type in notes_graph.edge_types.values():
                relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
            
            print("\nRelation type distribution:")
            for rel_type, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {rel_type}: {count} relations")
        
        print("===============================================")
        return notes_graph
    else:
        print("\n===== KNOWLEDGE GRAPH GENERATION FAILED =====")
        print(f"Files processed: {total_files} total, {successful} successful, {failed} failed")
        print("No valid graphs were generated.")
        print("===============================================")
        return None

if __name__ == "__main__":
    try:
        generate_obsidian_notes_kg()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()