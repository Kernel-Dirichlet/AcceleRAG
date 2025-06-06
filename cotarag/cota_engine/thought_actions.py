from abc import ABC, abstractmethod
from cotarag.accelerag.query_engines.query_engines import AnthropicEngine
import os


# Update ThoughtAction to include __str__ and __repr__ methods
class ThoughtAction(ABC):
    @abstractmethod
    def thought(self, input_data):
        pass

    @abstractmethod
    def action(self, thought_output):
        pass

    def __call__(self, input_data):

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
                 query_engine):
        
        assert query_engine is not None 
        self.query_engine = query_engine
    
    @abstractmethod
    def thought(self, input_data):
        pass
    
    @abstractmethod
    def action(self, thought_output):
        pass

class IterativeThoughtAction(ThoughtAction):

    @abstractmethod
    def thought(self):
        pass
    
    @abstractmethod
    def action(self):
        pass

    def __call__(self,
                 input_data,
                 max_iters = 100,
                 break_cond = None,
                 verbose = True):

        current_thought = input_data
        for i in range(max_iters):
            
            thought_output = self.thought(current_thought)
            action_output = self.action(thought_output)
            current_thought = action_output
            if verbose:
                print(f"step [{i+1}/{max_iters}]")
                print(f"thought: {current_thought} -> {action_output}")
            if break_cond is not None:
                if break_cond:
                    return {'thought': current_thought,
                            'max_iters': max_iters,
                            'iters': i+1}

        return {'thought': current_thought,
                'max_iters': max_iters,
                'iters': max_iters} #returns the current thought






