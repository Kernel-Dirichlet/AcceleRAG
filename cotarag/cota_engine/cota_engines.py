from .thought_actions import ThoughtAction

class CoTAEngine:
    def __init__(self, thought_actions):
        # Reasoning: Store the list of ThoughtActions
        self.thought_actions = thought_actions
        self.reasoning_chain = []  # Reasoning: List to track the outputs of each stage

    def run(self, initial_input):
        # Reasoning: Execute each ThoughtAction in sequence, passing the output of one as the input to the next
        current_output = initial_input
        for thought_action in self.thought_actions:
            try:
                # Reasoning: Execute the thought-action and track the output
                thought_output = thought_action.thought(current_output)
                action_output = thought_action.action(thought_output)
                self.reasoning_chain.append({
                    'input': current_output,
                    'thought_output': thought_output,
                    'action_output': action_output,
                    'query_engine': thought_action.__class__.__name__,
                    'args': thought_action.__dict__
                })
                current_output = action_output
            except Exception as e:
                # Reasoning: Track the error if a step fails
                self.reasoning_chain.append({
                    'input': current_output,
                    'error': str(e),
                    'query_engine': thought_action.__class__.__name__,
                    'args': thought_action.__dict__
                })
                raise
        return current_output

    def verify(self, initial_input):
        # Reasoning: Validate the outputs of each step in the CoTA chain
        current_output = initial_input
        for thought_action in self.thought_actions:
            current_output = thought_action(current_output)
            if current_output is None:
                raise ValueError(f"Output from {thought_action} is None.")
        return current_output

    def __str__(self):
        # Reasoning: Create a string representation of the CoTA chain
        chain = [f"input ({self.thought_actions[0].__class__.__name__})"]
        for thought_action in self.thought_actions:
            chain.append(f" -> ({thought_action})")
        return "".join(chain)

    def __repr__(self):
        # Reasoning: Use the same representation for string conversion
        return self.__str__()


