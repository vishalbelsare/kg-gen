# kg-gen: Knowledge Graph Generation from Any Text

📄 [**Paper**](https://arxiv.org/abs/2502.09956) | 🐍 [**Package**](https://pypi.org/project/kg-gen/) | 🤖 [**MCP**](https://github.com/stair-lab/kg-gen/tree/main/mcp/) | 🔬 [**Experiments**](https://github.com/stair-lab/kg-gen/tree/main/experiments/) | 👩🏻‍💻 [**Dataset**](https://huggingface.co/datasets/belindamo/wiki_qa_kggen) | 🐦 [**X Updates**](https://x.com/belindmo)

> 💡New! Try KGGen's visualizer at [kg-gen.org](https://kg-gen.org)

Welcome! `kg-gen` helps you extract knowledge graphs from any plain text using AI. It can process both small and large text inputs, and it can also handle messages in a conversation format.

Why generate knowledge graphs? `kg-gen` is great if you want to:
- Create a graph to assist with RAG (Retrieval-Augmented Generation)
- Create graph synthetic data for model training and testing
- Structure any text into a graph 
- Analyze the relationships between concepts in your source text

We support API-based and local model providers via [LiteLLM](https://docs.litellm.ai/docs/providers), including OpenAI, Ollama, Anthropic, Gemini, Deepseek, and others. We also use [DSPy](https://dspy.ai/) for structured output generation.

- Try it out by running the scripts in [`tests/`](https://github.com/stair-lab/kg-gen/tree/main/tests).
- Instructions to run our KG benchmark MINE are in [`MINE/`](https://github.com/stair-lab/kg-gen/tree/main/experiments/MINE).
- Read the paper: [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models](https://arxiv.org/abs/2502.09956)

## Powered by a model of your choice

Pass in a `model` string to use a model of your choice. Model calls are routed via LiteLLM, and usually LiteLLM goes by the format of `{model_provider}/{model_name}`. See specifically how to format it at [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).

Examples of models you can pass in:
- `openai/gpt-5`
- `gemini/gemini-2.5-flash`
- `ollama_chat/deepseek-r1:14b`

You may specify a custom API base url with `base_url` ([example here](https://github.com/stair-lab/kg-gen/tree/main/tests/test_custom_api_base.py)).

## Quick start

Install the module:
```bash
pip install kg-gen
```

Then import and use `kg-gen`. You can provide your text input in one of two formats:
1. A single string  
2. A list of Message objects (each with a role and content)

Below are some example snippets:
```python
from kg_gen import KGGen

# Initialize KGGen with optional configuration
kg = KGGen(
  model="openai/gpt-4o",  # Default model
  temperature=0.0,        # Default temperature
  api_key="YOUR_API_KEY"  # Optional if set in environment or using a local model
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
  input_data=text_input,
  context="Family relationships"
)
# Output: 
# entities={'Linda', 'Ben', 'Andrew', 'Josh'} 
# edges={'is brother of', 'is father of', 'is mother of'} 
# relations={('Ben', 'is brother of', 'Josh'), 
#           ('Andrew', 'is father of', 'Josh'), 
#           ('Linda', 'is mother of', 'Josh')}
```

### Visualizing KGs
```python
KGGen.visualize(graph, output_path, open_in_browser=True)
```

![viz-tool](images/viz-tool.png)

### More Examples - chunking, clustering, passing in a messages array 

```python
# EXAMPLE 2: Large text with chunking and clustering
with open('large_text.txt', 'r') as f:
  large_text = f.read()
  
# Example input text:
# """
# Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
# that uses multiple layers of neural networks. Supervised learning requires training data to learn
# patterns. Machine learning is a type of AI technology that enables computers to learn from data.
# AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
# Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
# many fields of study.
# ...
# """

graph_2 = kg.generate(
  input_data=large_text,
  chunk_size=5000,  # Process text in chunks of 5000 chars
  cluster=True      # Cluster similar entities and relations
)
# Output:
# entities={'neural networks', 'deep learning', 'machine learning', 'AI', 'artificial intelligence', 
#          'supervised learning', 'unsupervised learning', 'training data', ...} 
# edges={'is type of', 'requires', 'is subset of', 'uses', 'is related to', ...} 
# relations={('neural networks', 'is type of', 'machine learning'),
#           ('deep learning', 'is subset of', 'machine learning'),
#           ('supervised learning', 'requires', 'training data'),
#           ('machine learning', 'is type of', 'AI'),
#           ('AI', 'is related to', 'artificial intelligence'), ...}
# entity_clusters={
#   'artificial intelligence': {'AI', 'artificial intelligence'},
#   'machine learning': {'machine learning', 'ML'},
#   'neural networks': {'neural networks', 'neural nets', 'NN'}
#   ...
# }
# edge_clusters={
#   'is type of': {'is type of', 'is a type of', 'is a kind of'},
#   'is related to': {'is related to', 'is connected to', 'is associated with'
#  ...}
# }

# EXAMPLE 3: Messages array
messages = [
  {"role": "user", "content": "What is the capital of France?"}, 
  {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(input_data=messages)
# Output: 
# entities={'Paris', 'France'} 
# edges={'has capital'} 
# relations={('France', 'has capital', 'Paris')}

# EXAMPLE 4: Combining multiple graphs
text1 = "Linda is Joe's mother. Ben is Joe's brother."

# Input text 2: also goes by Joe."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

graph4_a = kg.generate(input_data=text1)
graph4_b = kg.generate(input_data=text2)

# Combine the graphs
combined_graph = kg.aggregate([graph4_a, graph4_b])

# Optionally cluster the combined graph
clustered_graph = kg.cluster(
  combined_graph,
  context="Family relationships"
)
# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Joe', 'Joseph', 'Judy'} 
# edges={'is mother of', 'is father of', 'is brother of', 'is sister of'} 
# relations={('Linda', 'is mother of', 'Joe'),
#           ('Ben', 'is brother of', 'Joe'),
#           ('Andrew', 'is father of', 'Joe'),
#           ('Judy', 'is sister of', 'Andrew')}
# entity_clusters={
#   'Joe': {'Joe', 'Joseph'},
#   ...
# }
# edge_clusters={ ... }
```

## Install from this repository:

Clone this repository and install dependencies using `pip install -e '.[dev]'`. 

You may verify that it works by running `python tests/test_basic.py` from the root directory. This will also generate a nice visualization in `tests/test_basic.html`.

### MCP Server for AI Agents

For AI agents that need persistent memory capabilities:

```bash
# Install and start MCP server
pip install kg-gen
kggen mcp

# Use with Claude Desktop, custom MCP clients, or other AI applications
```

See the [MCP Server documentation](mcp/) for detailed setup and integration instructions.


## Features

### Chunking Large Texts
For large texts, you can specify a `chunk_size` parameter to process the text in smaller chunks:
```python
graph = kg.generate(
  input_data=large_text,
  chunk_size=5000  # Process in chunks of 5000 characters
)
```

### Clustering Similar Entities and Relations
You can cluster similar entities and relations either during generation or afterwards:
```python
# During generation
graph = kg.generate(
  input_data=text,
  cluster=True,
  context="Optional context to guide clustering"
)

# Or after generation
clustered_graph = kg.cluster(
  graph,
  context="Optional context to guide clustering"
)
```

### Aggregating Multiple Graphs
You can combine multiple graphs using the aggregate method:
```python
graph1 = kg.generate(input_data=text1)
graph2 = kg.generate(input_data=text2)
combined_graph = kg.aggregate([graph1, graph2])
```

### Message Array Processing
When processing message arrays, kg-gen:
1. Preserves the role information from each message
2. Maintains message order and boundaries
3. Can extract entities and relationships:
   - Between concepts mentioned in messages
   - Between speakers (roles) and concepts
   - Across multiple messages in a conversation

For example, given this conversation:
```python
messages = [
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."}
]
```

The generated graph might include entities like:
- "user"
- "assistant" 
- "France"
- "Paris"

And relations like:
- (user, "asks about", "France")
- (assistant, "states", "Paris")
- (Paris, "is capital of", "France")


## License
The MIT License.
