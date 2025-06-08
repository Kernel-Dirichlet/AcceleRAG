from cotarag.cota_engine.thought_actions import LLMThoughtAction

class MetaCoT(LLMThoughtAction):
    """
    Meta Chain-of-Thought (MetaCoT) Prompting Pattern
    
    This pattern evaluates a chain of thought, scores each step and the chain itself,
    and optionally proposes a new improved chain:
    - Thought: Evaluates the chain of thought and scores each step
    - Action: Returns the evaluation scores and optionally proposes a new chain
    """
    
    def thought(self, input_data):
        # Reasoning: Evaluate the chain of thought and score each step
        query = input_data['query']
        chain = input_data['chain']
        self.query = query
        self.chain = chain
        evaluation_prompt = f"""Evaluate the following chain of thought for the query: {query}
Chain: {chain}
For each step, consider:
1. The difficulty of the step
2. If it logically follows from the previous step
3. Any possible reasoning flaws
Score each step and the chain itself on a scale of 1-10, where:
- 1: Hallucinatory, logically inconsistent, or impossible to implement
- 10: High quality, logically sound, and simple"""
        return {'thought': evaluation_prompt}
        
    def action(self, thought_output):
        # Reasoning: Use the evaluation prompt to generate scores and optionally propose a new chain
        evaluation = self.query_engine.generate_response(thought_output['thought'])
        scores = self.evaluate_chain(evaluation)
        return {
            'evaluation': evaluation,
            'scores': scores,
            'query': self.query,
            'chain': self.chain
        }

    def evaluate_chain(self, evaluation):
        # Reasoning: Parse the evaluation text to extract scores for each step and the entire chain
        # This method assumes the evaluation text is formatted with scores for each step and the chain
        # Example format: "Step 1: 8, Step 2: 7, Step 3: 9, Chain: 8"
        scores = {
            'chain_score': 0,  # Placeholder for the entire chain score
            'step_scores': []  # Placeholder for individual step scores
        }
        
        # Split the evaluation text into lines to process each step
        lines = evaluation.split('\n')
        for line in lines:
            if 'Step' in line:
                # Extract step number and score
                parts = line.split(':')
                if len(parts) == 2:
                    step_num = int(parts[0].split()[1])
                    score = int(parts[1].strip())
                    scores['step_scores'].append((step_num, score))
            elif 'Chain' in line:
                # Extract chain score
                parts = line.split(':')
                if len(parts) == 2:
                    scores['chain_score'] = int(parts[1].strip())
        
        return scores

    def propose_improved_chain(self, evaluation):
        # Reasoning: Propose a new improved chain based on the evaluation
        improvement_prompt = f"Based on the evaluation: {evaluation}, propose an improved chain of thought for the query: {self.query}"
        improved_chain = self.query_engine.generate_response(improvement_prompt)
        return improved_chain 
