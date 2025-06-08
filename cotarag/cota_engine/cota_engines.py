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

class ToolParser():
    """
    ToolParser class for evaluating and ordering tools based on a user request.
    
    This class takes a MetaCoT instance and a dictionary of tools, evaluates whether the tools can solve the user request,
    and outputs a list of tools in the order they should be applied, along with confidence scores.
    """
    
    def __init__(self, meta_cot, tools):
        # Reasoning: Initialize with a MetaCoT instance and a dictionary of tools
        self.meta_cot = meta_cot
        self.tools = tools

    def evaluate_tools(self, user_request):
        # Reasoning: Evaluate each tool against the user request
        tool_scores = []
        for tool_name, tool_info in self.tools.items():
            # Reasoning: Construct a prompt for evaluation
            evaluation_prompt = f"Evaluate if the tool '{tool_name}' can solve the following request: {user_request}\nTool Info: {tool_info.get('docstring', 'No description available')}"
            
            # Reasoning: Get evaluation from MetaCoT
            evaluation = self.meta_cot.run(evaluation_prompt)
            
            # Reasoning: Extract confidence score from evaluation
            try:
                # Reasoning: Look for a score in the evaluation text
                score_text = evaluation.split("Score:")[-1].strip()
                score = float(score_text.split()[0])
            except:
                # Reasoning: Default to low confidence if score can't be extracted
                score = 0.0
            
            tool_scores.append((tool_name, score))
        
        # Reasoning: Sort tools by confidence score
        return sorted(tool_scores, key=lambda x: x[1], reverse=True)
