from cotarag.cota_engine.thought_actions import LLMThoughtAction

class IOPrompt(LLMThoughtAction):
    """
    Simple I/O Prompting Pattern
    
    This is the most basic pattern where:
    - Thought: Just passes through the query
    - Action: Just returns the LLM's response (a "no-op")
    """
    
    def thought(self, input_data):
        # Reasoning: Simply return the query as the prompt
        return input_data['query']
        
    def action(self, thought_output):
        # Reasoning: Just return the LLM's response
        return {
            'response': self.query_engine.generate_response(thought_output)
        }

class ChainOfThought(LLMThoughtAction):
    """
    Chain-of-Thought (CoT) Prompting Pattern
    
    This pattern breaks down reasoning step-by-step:
    - Thought: Constructs a prompt asking for step-by-step reasoning
    - Action: Returns the LLM's response with the reasoning chain
    """
    
    def thought(self, query):
        # Reasoning: Store the query and construct a prompt asking for step-by-step reasoning
        self.query = query
        cot_header = """break down your reasoning for all subsequent queries step-by-step, output your steps as a chain 
                     step 1 -> step 2 -> ... -> step N \nQuery: """
        self.thought = f"{cot_header}:{query}"
        return {'thought': self.thought}

    def action(self, thought_input):
        # Reasoning: Return the LLM's response along with the original query and thought
        return {
            'action': self.query_engine.generate_response(thought_input['thought']),
            'query': self.query,
            'thought': self.thought
        }

    def output_mermaid(self, response):
        # Reasoning: Convert the response into a Mermaid diagram format
        steps = response.split(' -> ')
        mermaid = "graph TD\n"
        for i, step in enumerate(steps):
            mermaid += f"    step{i+1}[{step.strip()}]\n"
            if i < len(steps) - 1:
                mermaid += f"    step{i+1} --> step{i+2}\n"
        return mermaid

class FewShotPrompt(LLMThoughtAction):
    """
    Few-Shot Prompting Pattern
    
    This pattern formats a query with examples to guide the LLM's response:
    - Thought: Formats the query with provided examples
    - Action: Returns the LLM's response based on the formatted prompt
    """
    
    def thought(self, input_data):
        # Reasoning: Format the query with examples to guide the LLM
        query = input_data['query']
        examples = input_data['examples']
        formatted_prompt = f"Query: {query}\nExamples:\n{examples}"
        return {'thought': formatted_prompt}
        
    def action(self, thought_output):
        # Reasoning: Use the formatted prompt to generate a response from the LLM
        response = self.query_engine.generate_response(thought_output['thought'])
        return {'action': response}

class ChainOfThoughtSC(LLMThoughtAction):
    """
    Chain-of-Thought with Self-Consistency (CoT-SC) Prompting Pattern
    
    This pattern generates multiple reasoning chains and compares them to find the most consistent answer:
    - Thought: Constructs a prompt asking for multiple independent reasoning chains
    - Action: Returns the LLM's response with the reasoning chains and comparison
    """
    
    def thought(self, input_data, num_chains=3):
        # Reasoning: Construct a prompt asking for multiple independent reasoning chains
        query = input_data['query']
        cot_sc_header = """You are a careful and logical thinker
        For the following query, generate multiple independent reasoning chains
        (use {num_chains} in total) such that each chain is an attempt to answer the query correctly,
        using different logical approaches or perspectives if possible.
        
        After generating all reasoning chains, compare the final answers and choose
        the one that appears most frequently or is most logically sound.
        Query: {query}"""
        thought = cot_sc_header.format(query=query, num_chains=num_chains)
        return {'thought': thought}
        
    def action(self, thought_output):
        # Reasoning: Use the prompt to generate a response with multiple reasoning chains
        action = self.query_engine.generate_response(thought_output['thought'])
        return {'action': action}

    def output_mermaid(self, response):
        # Reasoning: Convert the response into a Mermaid diagram format
        chains = response.split('\n\n')
        mermaid = "graph TD\n"
        for i, chain in enumerate(chains):
            steps = chain.split(' -> ')
            for j, step in enumerate(steps):
                mermaid += f"    chain{i+1}step{j+1}[{step.strip()}]\n"
                if j < len(steps) - 1:
                    mermaid += f"    chain{i+1}step{j+1} --> chain{i+1}step{j+2}\n"
        return mermaid 
