from abc import ABC, abstractmethod
from accelerag.query_engines.query_engines import AnthropicEngine


# Update ThoughtAction to include __str__ and __repr__ methods
class ThoughtAction(ABC):
    @abstractmethod
    def thought(self, input_data):
        # Reasoning: This method will be implemented by subclasses to perform a thought process
        pass

    @abstractmethod
    def action(self, thought_output):
        # Reasoning: This method will be implemented by subclasses to perform an action based on the thought output
        pass

    def __call__(self, input_data):
        # Reasoning: Execute the thought process and then the action
        thought_output = self.thought(input_data)
        if thought_output is None:
            raise ValueError("Thought output cannot be None.")
        action_output = self.action(thought_output)
        if action_output is None:
            raise ValueError("Action output cannot be None.")
        return action_output

    def __str__(self):
        # Reasoning: Return the class name as the string representation
        return self.__class__.__name__

    def __repr__(self):
        # Reasoning: Use the same representation for string conversion
        return self.__str__()

class LLMThoughtAction(ThoughtAction):
    def __init__(self,
                 api_key = None,
                 query_engine = None):
        # Reasoning: Use the Anthropic API by default if no query_engine is provided
        if query_engine is None:
            query_engine = AnthropicEngine(api_key = api_key)
        self.query_engine = query_engine

    def thought(self, input_data):
        # Reasoning: Format the input into the query engine
        return self.query_engine.generate_response(input_data)

    def action(self, thought_output):
        # Reasoning: This method will be implemented by subclasses to perform an action based on the thought output
        pass 
    
    
