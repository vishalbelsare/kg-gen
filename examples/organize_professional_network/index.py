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
    
    # Create a custom implementation that avoids the Literal type issue
    try:
        # Extract entities and relations directly with DSPy to avoid the Literal type error
        import dspy
        
        # Initialize DSPy with the model
        dspy_module = kg_gen.dspy
        
        class CustomEntitiesWithTypes(dspy.Signature):
            """Extract key entities from the text and assign types to them.
            This is for building a professional network knowledge graph.
            Only use the allowed entity types."""
            
            text: str = dspy.InputField()
            allowed_types: list[str] = dspy.InputField(desc="List of allowed entity types")
            entities: list[dict] = dspy.OutputField(desc="List of entities with their types as {entity: str, type: str}")
        
        class CustomRelationsWithTypes(dspy.Signature):
            """Extract relationships between entities in the text.
            This is for building a professional network knowledge graph.
            Only use the allowed relationship types."""
            
            text: str = dspy.InputField()
            entities: list[dict] = dspy.InputField(desc="List of entities with their types")
            allowed_relation_types: list[str] = dspy.InputField(desc="List of allowed relationship types")
            relations: list[dict] = dspy.OutputField(desc="List of relations as {subject: str, predicate: str, object: str, type: str}")
        
        # Extract entities with types
        extract_entities = dspy.Predict(CustomEntitiesWithTypes)
        entities_result = extract_entities(
            text=content,
            allowed_types=ENTITY_TYPES
        )
        
        # Prepare entities for relation extraction
        entities = [item["entity"] for item in entities_result.entities]
        entity_types = {item["entity"]: item["type"] for item in entities_result.entities if "type" in item}
        
        # Skip if no entities found
        if not entities or len(entities) == 0:
            print(f"No entities found in {person_name}, skipping...")
            return None
        
        # Extract relations with types
        extract_relations = dspy.Predict(CustomRelationsWithTypes)
        relations_result = extract_relations(
            text=content,
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
        print(f"Error in custom implementation for {person_name}: {str(e)}")
        
        # Fallback to a simpler approach that doesn't use the Pydantic Literal type
        try:
            # Use a simpler approach without Pydantic models that use Literal
            content_with_context = f"""
Professional Network Knowledge Graph
-----------------------------------
Entity Types: {', '.join(ENTITY_TYPES)}
Relationship Types: {', '.join(EDGE_TYPES)}
-----------------------------------

{content}
"""
            # Generate a simple graph without clustering to avoid the error
            person_graph = kg_gen.generate(
                input_data=content_with_context,
                context="Professional networking and relationship notes. Extract entities and relationships for a professional network.",
                node_type=ENTITY_TYPES,
                edge_type=EDGE_TYPES,
                require_node_type=True,
                require_edge_type=True,
                cluster=False,  # Skip clustering to avoid additional errors
                output_folder=output_dir
            )
            
            return person_graph
            
        except Exception as e2:
            print(f"Both approaches failed for {person_name}: {str(e2)}")
            return None

def generate_professional_network_kg():
    """
    Generate knowledge graph from professional network markdown files
    """
    # Get all markdown files
    people_dir = os.path.join(os.path.dirname(__file__), "people")
    markdown_files = glob.glob(os.path.join(people_dir, "*.md"))
    
    # Process just the example file for testing
    example_file = os.path.join(people_dir, "example.md")
    if os.path.exists(example_file):
        print("Processing the example file for testing")
        markdown_files = [example_file]
    elif markdown_files:
        print(f"Found {len(markdown_files)} markdown files, processing first one for testing")
        markdown_files = markdown_files[:1]  # Start with just 1 file for testing
    else:
        print("No markdown files found in directory:", people_dir)
    
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
    try:
        generate_professional_network_kg()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()