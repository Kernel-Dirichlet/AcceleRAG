from .thought_actions import ThoughtAction, LLMThoughtAction


class CoTAEngine():
    def __init__(self, thought_actions):

        self.thought_actions = thought_actions
        self.reasoning_chain = []  

    def run(self, initial_input):
        
        current_input = initial_input
        for i,thought_action in enumerate(self.thought_actions):
            try:
                output = thought_action(current_input)
                state = {'input': current_input,
                         'output': output,
                         'thought_action': thought_action.__class__.__name__,
                         'args': thought_action.__dict__}

                self.reasoning_chain.append(state)
                current_input = output
            except Exception as e:
                print(f"WARNING! Error encountered: {str(e)}") 
                state = {'input': current_input,
                         'output': str(e),
                         'thought_action': thought_action.__class__.__name__,
                         'args': thought_action.__dict__}

                self.reasoning_chain.append(state)
                return state
        try:
            return output['action']
        except:
            return output

    def __str__(self):
        
        chain = [f"({self.thought_actions[0].__class__.__name__})"]
        for thought_action in self.thought_actions:
            chain.append(f" -> ({thought_action})")
        return "".join(chain)

    def __repr__(self):

        return self.__str__()
