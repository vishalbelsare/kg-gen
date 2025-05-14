from typing import List
import dspy 

def get_entities(dspy: dspy.dspy, input_data: str, is_conversation: bool = False, context: str = "") -> List[str]:

  class TextEntities(dspy.Signature):
    __doc__ = f"""Extract key entities from the source text. Extracted entities are subjects or objects. {context}
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""
    
    source_text: str = dspy.InputField()  
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

  class ConversationEntities(dspy.Signature):
    __doc__ = f"""Extract key entities from the conversation Extracted entities are subjects or objects. {context}
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""
    
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

  if is_conversation:
    extract = dspy.Predict(ConversationEntities)
  else:
    extract = dspy.Predict(TextEntities)
    
  result = extract(source_text=input_data)
  return result.entities

