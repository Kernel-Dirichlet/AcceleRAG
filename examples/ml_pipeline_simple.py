import os
import io
import json
import sys 
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import redirect_stdout
from cotarag.cota_engine.thought_actions import LLMThoughtAction

"""
ML Pipeline Thought-Action Implementation

This module demonstrates the power of the Thought-Action pattern for ML experimentation:

1. Thought-Action Pattern Overview:
   - Thought: Generates/processes information (like code or analysis)
   - Action: Executes the thought's output and produces results
   - The pattern separates planning (thought) from execution (action)
   - Each step's output feeds into the next, creating a chain of reasoning

2. Revolutionizing Hyperparameter Optimization:
   Traditional approaches like Bayesian Optimization or Genetic Algorithms:
   - Require many expensive iterations
   - Need careful tuning of their own parameters
   - Often get stuck in local optima
   - Don't understand the problem domain
   
   Thought-Action Approach:
   - Uses LLM to understand the problem and metrics
   - Makes informed decisions based on domain knowledge
   - Can explain its reasoning and recommendations
   - Adapts strategy based on results
   - Single prompt replaces complex optimization algorithms
   
   Why This Matters:
   - Efficiency: Fewer iterations needed
   - Transparency: Clear reasoning for each decision
   - Adaptability: Can handle non-standard metrics
   - Domain Awareness: Uses problem-specific knowledge
   - Cost Effective: No need for expensive optimization libraries

3. Why This Pattern Works Well for ML:
   - Separation of Concerns:
     * Thought: Handles parameter validation, code generation, and template formatting
     * Action: Manages execution, result collection, and LLM analysis
   - Chain of Reasoning:
     * Each iteration builds on previous results
     * LLM analysis informs next iteration's parameters
   - Error Isolation:
     * Errors in thought don't affect action execution
     * Each step can be debugged independently

4. Implementation Details:
   - Thought Method:
     * Takes input_data dictionary with parameters
     * Validates and formats parameters
     * Generates executable code template
   - Action Method:
     * Executes generated code safely
     * Captures metrics and results
     * Uses LLM to analyze results
     * Extracts hyperparameter recommendations

5. Benefits of This Approach:
   - Modularity: Each component can be modified independently
   - Reusability: Pattern can be applied to other ML tasks
   - Transparency: Clear separation between planning and execution
   - Extensibility: Easy to add new analysis or metrics
"""

class MLPipeline(LLMThoughtAction):
    def __init__(self, api_key=None, query_engine=None):
        # Reasoning: Initialize parent class for LLM capabilities
        super().__init__(api_key=api_key, query_engine=query_engine)
        
    def thought(self, input_data):
        # Reasoning: Extract parameters from input_data
        if not isinstance(input_data, dict):
            raise ValueError("input_data must be a dictionary with parameters")
            
        n_components = input_data.get('n_components')
        num_trees = input_data.get('num_trees')
        max_depth = input_data.get('max_depth')
        
        if any(x is None for x in [n_components, num_trees, max_depth]):
            raise ValueError("Missing required parameters: n_components, num_trees, max_depth")
            
        # Reasoning: Format the code template with provided arguments
        code_template = """
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# Reasoning: Load and prepare data
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reasoning: Create and execute pipeline
scaler = StandardScaler()
pca = PCA(n_components={n_components})
rf = RandomForestClassifier(n_estimators={num_trees}, max_depth={max_depth}, random_state=42)

# Reasoning: Fit and transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Reasoning: Train and predict
rf.fit(X_train_pca, y_train)
y_pred = rf.predict(X_test_pca)

# Reasoning: Calculate metrics
metrics = {{
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted'),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'feature_importance': rf.feature_importances_.tolist(),
    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    'current_params': {{
        'n_components': {n_components},
        'num_trees': {num_trees},
        'max_depth': {max_depth}
    }}
}}

# Reasoning: Store results for analysis
result = json.dumps(metrics, indent=2)
"""
        # Reasoning: Format the template with provided arguments
        try:
            formatted_code = code_template.format(
                n_components=n_components,
                num_trees=num_trees,
                max_depth=max_depth
            )
        except KeyError as e:
            raise ValueError(f"Missing required argument: {e}")
            
        return formatted_code
        
    def action(self, code):
        # Reasoning: Execute the code and capture output
        try:
            # Reasoning: Create a string buffer to capture stdout
            output_buffer = io.StringIO()
            
            # Reasoning: Execute code and capture output
            with redirect_stdout(output_buffer):
                # Reasoning: Execute in a new namespace to avoid pollution
                namespace = {}
                exec(code, namespace)
                
            # Reasoning: Get metrics as JSON string
            metrics_str = namespace.get('result', '{}')
            
            # Reasoning: Analyze results using LLM
            analysis_prompt = f"""Analyze the following ML pipeline results and suggest improvements:

Current Results:
{metrics_str}

Please provide your analysis in two parts:

PART 1 - Performance Analysis:
1. Performance Summary:
   - Overall accuracy and key metrics
   - Strengths and weaknesses
   - Class-wise performance (if applicable)

2. PCA Analysis:
   - Impact of current n_components
   - Explained variance insights

3. Random Forest Analysis:
   - Current tree configuration effectiveness
   - Feature importance insights

4. General Recommendations:
   - Potential model architecture changes
   - Additional features or preprocessing to consider

PART 2 - Hyperparameter Recommendations:
Format your hyperparameter recommendations exactly as follows:
max_depth = [suggested_value]
num_trees = [suggested_value]
n_components = [suggested_value]

Analysis:"""

            # Reasoning: Get LLM analysis
            analysis = self.query_engine.generate_response(analysis_prompt)
            
            # Reasoning: Extract hyperparameter recommendations
            hp_recommendations = {}
            for line in analysis.split('\n'):
                if '=' in line:
                    param, value = line.split('=')
                    param = param.strip()
                    value = value.strip()
                    if param in ['max_depth', 'num_trees', 'n_components']:
                        try:
                            hp_recommendations[param] = int(value)
                        except ValueError:
                            hp_recommendations[param] = value
            
            # Reasoning: Return metrics, analysis, and structured recommendations
            return {
                'metrics': json.loads(metrics_str),
                'analysis': analysis,
                'hp_recommendations': hp_recommendations
            }
            
        except Exception as e:
            # Reasoning: Handle execution errors
            error_msg = f"Error executing ML pipeline: {str(e)}"
            return {
                'error': error_msg,
                'analysis': None,
                'hp_recommendations': None
            }

def main():
    # Reasoning: Check for API key
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    print("\n=== ML Pipeline Thought-Action Demo ===")
    
    # Reasoning: Create pipeline with initial parameters
    print("\n1. Initializing ML Pipeline...")
    pipeline = MLPipeline(api_key=api_key)
    
    # Reasoning: First iteration with default parameters
    print("\n2. Running first iteration...")
    print("   Parameters:")
    print("   - n_components: 10")
    print("   - num_trees: 100")
    print("   - max_depth: 5")
    
    result1 = pipeline(input_data={
        'n_components': 10,
        'num_trees': 100,
        'max_depth': 5
    })
    
    # Reasoning: Print first iteration results
    print("\n3. First Iteration Results:")
    print("   Metrics:")
    metrics = result1['metrics']
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - F1 Score: {metrics['f1']:.4f}")
    
    print("\n   Analysis:")
    print(result1['analysis'])
    
    # Reasoning: Get hyperparameter recommendations
    hp_recs = result1['hp_recommendations']
    print("\n   Recommended Parameters for Next Iteration:")
    print(f"   - n_components: {hp_recs['n_components']}")
    print(f"   - num_trees: {hp_recs['num_trees']}")
    print(f"   - max_depth: {hp_recs['max_depth']}")
    
    # Reasoning: Second iteration with recommended parameters
    print("\n4. Running second iteration with recommended parameters...")
    result2 = pipeline(input_data={
        'n_components': hp_recs['n_components'],
        'num_trees': hp_recs['num_trees'],
        'max_depth': hp_recs['max_depth']
    })
    
    # Reasoning: Print second iteration results
    print("\n5. Second Iteration Results:")
    print("   Metrics:")
    metrics = result2['metrics']
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - F1 Score: {metrics['f1']:.4f}")
    
    print("\n   Analysis:")
    print(result2['analysis'])
    
    # Reasoning: Compare iterations
    print("\n6. Performance Comparison:")
    print(f"   First Iteration Accuracy: {result1['metrics']['accuracy']:.4f}")
    print(f"   Second Iteration Accuracy: {result2['metrics']['accuracy']:.4f}")
    improvement = (result2['metrics']['accuracy'] - result1['metrics']['accuracy']) * 100
    print(f"   Improvement: {improvement:+.2f}%")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main() 
