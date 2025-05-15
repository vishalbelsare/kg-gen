def alphabetize_kg_json(kg_json):
    """Alphabetize entities, relations and edges in a knowledge graph JSON."""
    if "entities" in kg_json:
        kg_json["entities"] = sorted(kg_json["entities"])
        
    if "edges" in kg_json:
        kg_json["edges"] = sorted(kg_json["edges"])
        
    if "relations" in kg_json:
        # Sort relations by first element, then second, then third
        kg_json["relations"] = sorted(kg_json["relations"], 
                                key=lambda x: (x[0], x[1], x[2]))
    
    return kg_json

if __name__ == "__main__":
    import json
    import os

    article_dirs = [
        "tests/data/wiki_qa/articles_1",
        "tests/data/wiki_qa/articles_40k_ch", 
        "tests/data/wiki_qa/articles_400k_ch",
        "tests/data/wiki_qa/articles_4m_ch",
        "tests/data/wiki_qa/articles_20m_ch",
        "tests/data/wiki_qa/articles",
        "tests/data/wiki_qa/articles_w_context"
    ]

    for article_dir in article_dirs:
        # Construct path to aggregated KG file
        kg_dir = article_dir + "_kg"
        base_dir = "tests/data/wiki_qa/aggregated"
        kg_filename = os.path.basename(kg_dir) + ".json"
        kg_path = os.path.join(base_dir, kg_filename)

        if os.path.exists(kg_path):
            print(f"Processing {kg_path}")
            
            # Read JSON
            with open(kg_path, 'r') as f:
                kg_json = json.load(f)
            
            # Alphabetize
            kg_json = alphabetize_kg_json(kg_json)
            
            # Write back
            with open(kg_path, 'w') as f:
                json.dump(kg_json, f, indent=4)
                
            print(f"Saved alphabetized KG to {kg_path}")
        else:
            print(f"File not found: {kg_path}")
