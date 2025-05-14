from src.kg_gen import KGGen
import os
from dotenv import load_dotenv

def test_context_generation():
  # Load environment variables
  load_dotenv()
  
  # Initialize KGGen with temperature 0
  kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)
  
  # Load fresh wiki content
  with open('tests/data/fresh_wiki_article.md', 'r', encoding='utf-8') as f:
    text = f.read()
  
  # Generate graph without context
  graph_no_context = kg.generate(
    input_data=text,
    model="openai/gpt-4o",
    cluster=True
  )
  
  # Generate graph with context
  extraction_context = "This is about historical events and their key figures. Focus on extracting important dates, locations, and causal relationships."
  cluster_context = "Group related historical events, figures, and concepts together. Consider temporal and geographical relationships."
  graph_with_context = kg.generate(
    input_data=text,
    model="openai/gpt-4o",
    extraction_context=extraction_context,
    cluster_context=cluster_context,
    cluster=True
  )
  
  # Compare text results
  print("\nText Input Results:")
  print("Without context:")
  print("- Entities:", graph_no_context.entities)
  print("- Relations:", graph_no_context.relations)
  print("\nWith context:")
  print("- Entities:", graph_with_context.entities)
  print("- Relations:", graph_with_context.relations)
  
  # Test chat input with more complex conversation
  chat = [
    {"role": "user", "content": "Can you tell me about the key events and figures in this historical period?"},
    {"role": "assistant", "content": "I'll analyze the major historical events and their significance. Let me break down the key figures and their roles."},
    {"role": "user", "content": "What were the main causes and effects of these events?"},
    {"role": "assistant", "content": "The events were primarily driven by political and social factors. The outcomes had lasting impacts on subsequent developments."}
  ]
  
  # Generate chat graphs with and without context
  chat_graph_no_context = kg.generate(
    input_data=chat,
    model="openai/gpt-4o",
    cluster=True
  )
  chat_graph_with_context = kg.generate(
    input_data=chat,
    model="openai/gpt-4o", 
    extraction_context="This is a conversation about programming languages",
    cluster_context="Group programming language concepts",
    cluster=True
  )
  
  # Compare chat results
  print("\nChat Input Results:")
  print("Without context:")
  print("- Entities:", chat_graph_no_context.entities)
  print("- Relations:", chat_graph_no_context.relations)
  print("\nWith context:")
  print("- Entities:", chat_graph_with_context.entities)
  print("- Relations:", chat_graph_with_context.relations)
  
  # Test chunking
  long_text = text * 10  # Repeat text to make it longer
  
  # Generate chunked graphs with and without context
  chunked_graph_no_context = kg.generate(
    input_data=long_text,
    model="openai/gpt-4o",
    chunk_size=100,
    cluster=True
  )
  
  chunked_graph_with_context = kg.generate(
    input_data=long_text,
    model="openai/gpt-4o",
    extraction_context=extraction_context,
    cluster_context=cluster_context,
    chunk_size=100,
    cluster=True
  )
  
  # Compare chunked results
  print("\nChunked Input Results:")
  print("Without context:")
  print("- Entities:", chunked_graph_no_context.entities)
  print("- Relations:", chunked_graph_no_context.relations)
  print("\nWith context:")
  print("- Entities:", chunked_graph_with_context.entities)
  print("- Relations:", chunked_graph_with_context.relations)

if __name__ == "__main__":
  test_context_generation()
