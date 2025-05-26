import os
import unittest
import sys
import tempfile
import shutil

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cotarag.cota_engine.cota_engines import CoTAEngine
from cotarag.cota_engine.thought_actions import LLMThoughtAction

class TestCoTAEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reasoning: Get the API key from environment variables
        try:
            cls.api_key = os.environ.get("CLAUDE_API_KEY")
        except:
            cls.api_key = os.environ.get("OPENAI_API_KEY")
        if not cls.api_key:
            raise EnvironmentError("API_KEY must be set in environment variables")
        
        # Reasoning: Read API key from file if it's a file path
        if os.path.isfile(cls.api_key):
            with open(cls.api_key, 'r') as f:
                cls.api_key = f.read().strip()

        # Reasoning: Create a temporary folder for the mock codebase
        cls.mock_codebase_dir = tempfile.mkdtemp()
        cls.create_mock_codebase(cls.mock_codebase_dir)

    @classmethod
    def tearDownClass(cls):
        # Reasoning: Clean up the temporary directory after tests
        shutil.rmtree(cls.mock_codebase_dir)

    @staticmethod
    def create_mock_codebase(directory):
        # Reasoning: Create two different hello_world Python files in the mock codebase
        with open(os.path.join(directory, 'hello_world1.py'), 'w') as f:
            f.write('print("Hello, World!")')
        with open(os.path.join(directory, 'hello_world2.py'), 'w') as f:
            f.write('print("Hello, World!")')

    def test_llm_thought_action_chain(self):
        # Reasoning: Read the contents of the hello_world files
        with open(os.path.join(self.mock_codebase_dir, 'hello_world1.py'), 'r') as f:
            hello_world1_content = f.read()
        with open(os.path.join(self.mock_codebase_dir, 'hello_world2.py'), 'r') as f:
            hello_world2_content = f.read()

        # Reasoning: Format the prompt for the first LLMThoughtAction
        code_prompt = hello_world1_content + "\n" + hello_world2_content + "\nSummarize the code."

        # Reasoning: Define the first LLMThoughtAction to summarize the code
        class EvaluateCodebaseAction(LLMThoughtAction):
            def action(self, thought_output):
                # Write the summary to codebase_summary
                with open('codebase_summary', 'w') as f:
                    f.write(thought_output)
                return thought_output

        # Reasoning: Run the first step to get the summary
        eval_action = EvaluateCodebaseAction(api_key=self.api_key)
        summary = eval_action.thought(code_prompt)
        eval_action.action(summary)

        # Reasoning: Read the summary from the file for the next step
        with open('codebase_summary', 'r') as f:
            summary_content = f.read()

        # Reasoning: Format the prompt for the second LLMThoughtAction
        improvement_prompt = f"Suggest improvements for the following code summary:\n{summary_content}"

        # Reasoning: Define the second LLMThoughtAction to suggest improvements
        class AnalyzeCodebaseAction(LLMThoughtAction):
            def action(self, thought_output):
                # Write the improvements to codebase_TODOs
                with open('codebase_TODOs', 'w') as f:
                    f.write(thought_output)
                return thought_output

        # Reasoning: Run the second step to get the improvements
        analyze_action = AnalyzeCodebaseAction(api_key=self.api_key)
        improvements = analyze_action.thought(improvement_prompt)
        analyze_action.action(improvements)

        # Reasoning: Verify that the files were created
        self.assertTrue(os.path.exists('codebase_summary'))
        self.assertTrue(os.path.exists('codebase_TODOs'))

        # Reasoning: Print out their contents for inspection
        if os.path.exists('codebase_summary'):
            with open('codebase_summary', 'r') as f:
                summary_content = f.read()
            print("Contents of codebase_summary:")
            print(summary_content)
            print("-" * 50)

        if os.path.exists('codebase_TODOs'):
            with open('codebase_TODOs', 'r') as f:
                todos_content = f.read()
            print("Contents of codebase_TODOs:")
            print(todos_content)
            print("-" * 50)

        # Reasoning: Print the ASCII chain of thought-action steps using __str__ methods
        print("ASCII Chain of Thought-Action (CoTA) Steps:")
        print(f"input -> {EvaluateCodebaseAction().__str__()} -> {AnalyzeCodebaseAction().__str__()}")

if __name__ == '__main__':
    unittest.main() 
