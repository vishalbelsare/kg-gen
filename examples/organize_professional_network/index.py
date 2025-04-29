"""
Professional Network Knowledge Graph Generator

This script analyzes markdown files containing meeting notes and professional 
interactions to generate a knowledge graph of professional relationships.
"""
import os
import glob
from typing import List, Dict, Tuple
from src.kg_gen import KGGen

# Define entity and edge types for professional networking
ENTITY_TYPES = [
  "Person",       # An individual contact in the professional network
  "Company",      # An organization or business entity
  "Role",         # A professional position or title (e.g., "CEO", "Software Engineer")
  "Skill",        # A professional ability or expertise area
  "Project",      # A specific initiative or work engagement
  "Event",        # A meeting, conference, or other professional gathering
  "Industry",     # A business sector or field
  "Location"      # A geographical place or office location
]

EDGE_TYPES = [
  "works_at",          # Person → Company relationship
  "knows",             # Person → Person acquaintance relationship
  "skilled_in",        # Person → Skill expertise relationship
  "interested_in",     # Person → Topic/Industry interest relationship
  "collaborated_with", # Person → Person working relationship
  "attended",          # Person → Event participation relationship
  "located_in",        # Person/Company → Location geographical relationship
  "manages",           # Person → Person supervisory relationship
  "reports_to",        # Person → Person hierarchical relationship
  "part_of",           # Any entity that belongs to a larger entity/group
  "leads",             # Person → Project/Team leadership relationship
  "mentioned",         # Person mentioned a topic in conversation
  "recommended",       # Person recommended something (skill, person, etc.)
  "connected_through"  # Indirect relationship via mutual connection
]

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
        person_name = os.path.splitext(filename)[0]
        
        with open(file_path, 'r') as f:
            content = f.read()
            files_dict[person_name] = content
            
    return files_dict

def process_single_file(file_path: str):
    """
    Process a single markdown file to generate and save a knowledge graph
    
    Args:
        file_path: Path to the markdown file
    """
    # Initialize KGGen
    kg_gen = KGGen(model="openai/gpt-4o", temperature=0.0)
    
    # Get person name from file
    filename = os.path.basename(file_path)
    person_name = os.path.splitext(filename)[0]
    print(f"Processing {person_name}...")
    
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "output", person_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # First extract entities only to avoid empty literal issue
    import dspy
    from src.kg_gen.steps._1_get_entities import get_entities
    
    # Initialize DSPy
    dspy_module = kg_gen.dspy
    
    # Get entities first with strict typing
    entities_result = get_entities(
        dspy_module,
        content,
        is_conversation=False,
        node_types=ENTITY_TYPES,
        require_node_type=True  # Enforce that all extracted entities must have a type from ENTITY_TYPES
    )
    
    # Skip if no entities found
    if not entities_result or len(entities_result) == 0:
        print(f"No entities found in {person_name}, skipping...")
        return None
    
    # Now generate the full graph with the extracted entities, enforcing strict node and edge typing
    person_graph = kg_gen.generate(
        input_data=content,
        context="Professional networking and relationship notes. Strictly enforce all entity and relationship types.",
        node_type=ENTITY_TYPES,  # Restrict to only these node types
        edge_type=EDGE_TYPES,    # Restrict to only these edge types
        require_node_type=True,  # Every entity MUST have one of the specified types
        require_edge_type=True,  # Every relation MUST have one of the specified types
        cluster=True,
        output_folder=output_dir
    )
    
    return person_graph

def generate_professional_network_kg():
    """
    Generate knowledge graph from professional network markdown files
    """
    # Get all markdown files
    people_dir = os.path.join(os.path.dirname(__file__), "people")
    markdown_files = glob.glob(os.path.join(people_dir, "*.md"))
    
    # Limit to a few files for testing
    markdown_files = markdown_files[:2]  # Start with just 2 files for testing
    
    # Process each file individually
    all_graphs = []
    for file_path in markdown_files:
        try:
            graph = process_single_file(file_path)
            if graph:
                all_graphs.append(graph)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Only aggregate if we have graphs
    if all_graphs:
        # Initialize KGGen for aggregation
        kg_gen = KGGen(model="openai/gpt-4o", temperature=0.0)
        
        # Aggregate all individual graphs into a combined network graph
        network_graph = kg_gen.aggregate(all_graphs)
        
        # Save the aggregated graph
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a comprehensive representation of the graph for saving
        import json
        
        # Build a more detailed graph representation with typing information
        graph_dict = {
            'entities': list(network_graph.entities),
            'relations': [list(r) for r in network_graph.relations],
            'edges': list(network_graph.edges),
        }
        
        # Include entity type information if available
        if hasattr(network_graph, 'entity_types') and network_graph.entity_types:
            graph_dict['entity_types'] = network_graph.entity_types
        
        # Include edge type information if available
        if hasattr(network_graph, 'edge_types') and network_graph.edge_types:
            graph_dict['edge_types'] = network_graph.edge_types
            
        # Include clustering information if available
        if hasattr(network_graph, 'entity_clusters') and network_graph.entity_clusters:
            graph_dict['entity_clusters'] = {rep: list(cluster) for rep, cluster in network_graph.entity_clusters.items()}
        if hasattr(network_graph, 'edge_clusters') and network_graph.edge_clusters:
            graph_dict['edge_clusters'] = {rep: list(cluster) for rep, cluster in network_graph.edge_clusters.items()}
        
        with open(os.path.join(output_dir, 'network_graph.json'), 'w') as f:
            json.dump(graph_dict, f, indent=2)
        
        print(f"Professional network knowledge graph generated in {output_dir}")
        print(f"Total entities: {len(network_graph.entities)}")
        print(f"Total relations: {len(network_graph.relations)}")
        
        return network_graph
    else:
        print("No valid graphs were generated.")
        return None

if __name__ == "__main__":
    generate_professional_network_kg()