from ..models import Graph
import dspy
from typing import Optional, Literal
from pydantic import BaseModel

LOOP_N = 8 
BATCH_SIZE = 10
ItemType = Literal["entities", "edges"]

class ChooseRepresentative(dspy.Signature):
  """Select the best item name to represent the cluster, ideally from the cluster.
  Prefer shorter names and generalizability across the cluster."""
  
  cluster: list[str] = dspy.InputField()
  context: str = dspy.InputField(desc="the larger context in which the items appear")
  representative: str = dspy.OutputField()

choose_rep = dspy.Predict(ChooseRepresentative)

class Cluster(BaseModel):
  representative: str
  members: set[str]  # Changed from list to set for consistency

def extract_cluster(items: set[str], context: str) -> set[str]:
  items_str = "\n- ".join(items)
  
  class ExtractCluster(dspy.Signature):
    __doc__ = f"""Find one cluster of related items from the list.
    A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases. 
    Return populated list only if you find exact-match items that clearly belong together, else return empty list.
    List: {items_str}"""
    
    items: list[str] = dspy.InputField()
    context: str = dspy.InputField(desc="The larger context in which the items appear")
    cluster: list[str] = dspy.OutputField(desc="Exact-match items that belong together")

  extract = dspy.Predict(ExtractCluster)
  
  cluster: set[str] = set(extract(
    items=list(items), 
    context=context
  ).cluster)
  return {item for item in cluster if item in items}
    
def validate_cluster(cluster: set[str], context: str) -> set[str]:
  cluster_str = "\n- ".join(cluster)
      
  class ValidateCluster(dspy.Signature):
    __doc__ = f"""Validate if these items belong in the same cluster, given the context.
    A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases. 
    Return populated list only if you find items that clearly belong together, else return empty list.
    Cluster items: {cluster_str}"""
    
    cluster: list[str] = dspy.InputField()
    context: str = dspy.InputField(desc="The larger context in which the items appear")
    validated_items: list[str] = dspy.OutputField(desc="All the exact-match items that belong together in the cluster")
  
  validate = dspy.Predict(ValidateCluster)
  
  validated_cluster: set[str] = set(validate(
    cluster=list(cluster), 
    context=context
  ).validated_items)
  return {item for item in validated_cluster if item in cluster}  

def cluster_items(dspy: dspy.dspy, items: set[str], item_type: ItemType = "entities", context: str = "") -> tuple[set[str], dict[str, set[str]]]:
  """
  Clusters similar items and returns:
  1. A set containing all unique representatives
  2. A dictionary mapping a representative to a set of members
  """
  
  context = f"{item_type} of a graph extracted from source text." + context
  remaining_items = items.copy()
  clusters: list[Cluster] = []
  no_progress_count = 0
  
  while len(remaining_items) > 0:
      
    suggested_cluster: set[str] = extract_cluster(remaining_items, context)
    
    if len(suggested_cluster) > 0:
      validated_cluster: set[str] = validate_cluster(suggested_cluster, context)
      
      if len(validated_cluster) > 1:
        no_progress_count = 0
        
        representative = choose_rep(
          cluster=list(validated_cluster), 
          context=context
        ).representative
        
        clusters.append(Cluster(
          representative=representative,
          members=validated_cluster
        ))
        remaining_items = {item for item in remaining_items if item not in validated_cluster}
        continue
      
    no_progress_count += 1
    
    if no_progress_count >= LOOP_N or len(remaining_items) == 0:
      break
    
  class CheckExistingClusters(dspy.Signature):
    """Determine if the given items can be added to any of the existing clusters.
    Return representative of matching cluster for each item, or None if there is no match."""
    
    items: list[str] = dspy.InputField()
    clusters: list[Cluster] = dspy.InputField(desc="Mapping of cluster representatives to their cluster members")
    context: str = dspy.InputField(desc="The larger context in which the items appear")
    cluster_reps_that_items_belong_to: list[Optional[str]] = dspy.OutputField(desc="Ordered list of cluster representatives where each is the cluster where that item belongs to, or None if no match. THIS LIST LENGTH IS SAME AS ITEMS LIST LENGTH")

  check_existing = dspy.ChainOfThought(CheckExistingClusters)

  if len(remaining_items) > 0:
    items_to_process = list(remaining_items) 
      
    for i in range(0, len(items_to_process), BATCH_SIZE):
      batch = items_to_process[i:min(i + BATCH_SIZE, len(items_to_process))]
      
      if not clusters:
        for item in batch:
          clusters.append(Cluster(
            representative=item,
            members={item}  # Using set instead of list
          ))
        continue
      
      c_result = check_existing(
        items=batch,
        clusters=clusters,
        context=context
      )
      cluster_reps = c_result.cluster_reps_that_items_belong_to  
      
      cluster_map = {c.representative: c for c in clusters}
      
      item_assignments: dict[str, Optional[str]] = {} 
      
      for i, item in enumerate(batch):
        item_assignments[item] = None 
        
        rep = cluster_reps[i] if i < len(cluster_reps) else None
        
        target_cluster = None
        if rep is not None and rep in cluster_map:
            target_cluster = cluster_map[rep]

        if target_cluster:
          if item == target_cluster.representative or item in target_cluster.members:
              item_assignments[item] = target_cluster.representative 
              continue 

          potential_new_members: set[str] = target_cluster.members | {item}  # Already a set
          try:
              validated_items = validate_cluster(potential_new_members, context)

              if item in validated_items and len(validated_items) == len(potential_new_members):
                item_assignments[item] = target_cluster.representative 

          except Exception as e:
              print(f"Validation failed for item '{item}' potentially belonging to cluster '{target_cluster.representative}': {e}")

      new_cluster_items = set()  # Using set as intended
      for item, assigned_rep in item_assignments.items():
          if assigned_rep is not None:
              if assigned_rep in cluster_map:
                  cluster_map[assigned_rep].members.add(item)  # Using set.add() instead of list.append()
              else:
                  print(f"Error: Assigned representative '{assigned_rep}' not found in cluster_map for item '{item}'. Creating new cluster.")
                  if item not in cluster_map: 
                     new_cluster_items.add(item)  # Using set.add()
          else:
              if item not in cluster_map:
                   new_cluster_items.add(item)  # Using set.add()

      # Create the new Cluster objects for items that couldn't be assigned
      for item in new_cluster_items:
          if item not in cluster_map: 
              new_cluster = Cluster(representative=item, members={item})  # Using set
              clusters.append(new_cluster)
              cluster_map[item] = new_cluster # Update map for internal consistency

  # No need to convert to set here since members are already sets
  final_clusters_dict = {c.representative: c.members for c in clusters}
  new_items = set(final_clusters_dict.keys()) 
  
  return new_items, final_clusters_dict

def cluster_graph(dspy: dspy.dspy, graph: Graph, context: str = "") -> Graph:
  """Cluster entities and edges in a graph, updating relations accordingly.
  
  Args:
      dspy: The DSPy runtime
      graph: Input graph with entities, edges, and relations
      context: Additional context string for clustering
      
  Returns:
      Graph with clustered entities and edges, updated relations, and cluster mappings
  """
  entities, entity_clusters = cluster_items(dspy, graph.entities, "entities", context)
  edges, edge_clusters = cluster_items(dspy, graph.edges, "edges", context)
  
  # Update relations based on clusters
  relations: set[tuple[str, str, str]] = set()
  for s, p, o in graph.relations:
    # Look up subject in entity clusters
    if s not in entities:
      for rep, cluster in entity_clusters.items():
        if s in cluster:
          s = rep
          break
          
    # Look up predicate in edge clusters
    if p not in edges:
      for rep, cluster in edge_clusters.items():
        if p in cluster:
          p = rep
          break
          
    # Look up object in entity clusters
    if o not in entities:
      for rep, cluster in entity_clusters.items():
        if o in cluster:
          o = rep
          break
          
    relations.add((s, p, o))

  return Graph(
    entities=entities,  
    edges=edges,  
    relations=relations,
    entity_clusters=entity_clusters,
    edge_clusters=edge_clusters
  )

if __name__ == "__main__":
  import os
  from ..kg_gen import KGGen
  
  model = "openai/gpt-4o"
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)
    
  # Example with family relationships
  kg_gen = KGGen(
    model=model,
    temperature=0.0,
    api_key=api_key
  )
  graph = Graph(
    entities={
      "Linda", "Joshua", "Josh", "Ben", "Andrew", "Judy"
    },
    edges={
      "is mother of", "is brother of", "is father of",
      "is sister of", "is nephew of", "is aunt of",
      "is same as"
    },
    relations={
      ("Linda", "is mother of", "Joshua"),
      ("Ben", "is brother of", "Josh"),
      ("Andrew", "is father of", "Josh"),
      ("Judy", "is sister of", "Andrew"),
      ("Josh", "is nephew of", "Judy"),
      ("Judy", "is aunt of", "Josh"),
      ("Josh", "is same as", "Joshua")
    }
  )
  
  try: 
    clustered_graph = kg_gen.cluster(graph=graph)
    print('Clustered graph:', clustered_graph)
    
  except Exception as e:
    raise ValueError(e)