## Code-only for the Agent built in ./BuildYourOwnAgent.ipynb
"""
This Python File contains the Code "ONLY" for the custom implementation of the Agent. 
Use this File to build and iterate better.

Version 1.0 - More iterations are in the work, improving our Agent Step-by-Step.

What does this contain?
- A basic intuition of what Agents are? 
- An end-to-end skeletal structure of how Agents work.

What this does not contain?
- Agent's Memory
- Feedback and looping mechanism
    *This is set for future versions, where I would be explain in detail*
"""

## === Imports === 
import os
import requests
from dotenv import load_dotenv

# === Loading the API Key === 

load_dotenv()  # Load environment variables from a .env file
api_key = os.getenv("GEMINI_API_KEY")

# === Initializing the Gemini Client === 
"""Custom Implementation"""
class GeminiClient:
	def __init__(self, api_key: str):
		self.api_key = api_key
		self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
		self.headers = {
			"x-goog-api-key": self.api_key,
			"Content-Type": "application/json"
			}
	def generate_content(self, prompt) -> str:
		payload = {
			"contents": [
					{ "parts":[{"text":prompt}]}
				]
		}
		response = requests.post(self.endpoint, headers= self.headers, json = payload)
		if response.status_code ==200:
			data = response.json()
			try:
				return data['candidates'][0]['content']['parts'][0]['text']
			except (KeyError, IndexError):
				return "No response text found"
		else:
			return f"Error: {response.status_code} - {response.text}"

# Initializing client:
client = GeminiClient(api_key)

# === Tool Registry and Associated Tools === 
# Tool Registry:
class Tool:
    def __init__(self):
        self.tools = {}

    def registerTool(self, name=None, description=None):
        """Records a Tool"""
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = {
                "function_name": func,
                "description": description or func.__doc__,
                "call": lambda *args, **kwargs: func(*args, **kwargs)
            }
            return func
        return decorator

    def call(self, tool_name: str, *args, **kwargs):
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' does not exist.")
        return self.tools[tool_name]["call"](*args, **kwargs)

    def list_tools(self):
        return {name: meta["description"] for name, meta in self.tools.items()}

# Creating an instance of the Tool Registry:
toolFamily = Tool()

# === Building the Agent === 
# First, we would want our Agent to be a seperate object to make things easier:
class Agent():
  def __init__(self, work, toolfamily, llm_client):
    self.work = work
    self.toolfamily = toolfamily
    self.thought_prompt = self._buildThoughtPrompt()
    self.llm = llm_client

  def _buildThoughtPrompt(self) -> str:
    agentPrompt = f"You are an AI-system equipped with some tools. You are an expert {self.work}"
    outputGeneration = """
      You are expected to plan and reason the steps that need to be taken in order to arrive at the expected answer.
      Each step of your thinking should be shown as a dictionary element contained within a Python list.
      Each step should only contain the work related to the tool and nothing else.
      If the query can be resolved within a single step- go for it; Else document each step in the format given below.
      ```python
      [
        {
          "thought": "a-one-sentence-description-of-why-you-selected-this-tool",
          "action": (Tool_Name, *args, **kwargs)
        },
        ...
      ]
      ```
      Only generate this python object- do not generate any extra code or explainatory text
      Here are your Tools.
    """
    agentPrompt += outputGeneration
    for key, val in self.toolfamily.list_tools().items():
      samp = f"\nTool Name:\t{key}\nTool Description:\n{val}\n"
      agentPrompt += samp
    return agentPrompt

  def _think(self, query) -> list:
    """Runs the Thinking Step of the Agent"""
    prompt = f"{self.thought_prompt}\n\nHere is your Query {query}"
    answer = self.llm.generate_content(prompt)
    answer = answer.replace("python", "").replace("```", "")
    return eval(answer)

  def _act(self, thought: list[dict]) -> list:
      results = []
      for step in thought:
          action_tuple = step.get("action")
          if not action_tuple:
              continue
          tool_name, *args = action_tuple
          tool_fn = self.toolfamily.tools[tool_name]["function_name"]

          # Handle case: single argument which is a dict → unpack as kwargs
          if len(args) == 1 and isinstance(args[0], dict):
              result = tool_fn(**args[0])
          else:
              result = tool_fn(*args)

          results.append(result)
      return results
  def _observe(self, query, thought, data):
    prompt = f"""
      You are a finance agent- and you have performed certain operations.
      The query that was given to you was:
        {query}
      The Thought you processed was:
        {thought}
      The Data You recieved was:
        {data}
      Now, combine these information and give a suitable reply for the user.
      Notes: Give Smart financial answers related to the query-- and your answer should satisfy the user's question.
    """
    answer = self.llm.generate_content(prompt)
    return answer

  def _run_normal(self, query):
      thoughts = self._think(query)
      action = self._act(thoughts)
      observation = self._observe(query, thoughts, action)
      return observation

  def _run_verbose(self, query):
    print("Thought Prompt:\n", self.thought_prompt)
    print("Thinking...")
    thoughts = self._think(query)
    print(f"\n\nThoughts\n:{thoughts}")
    print("Gathering Information...")
    action = self._act(thoughts)
    print(f"\n\nInformation Gathered:\n{action}")
    print("\n\n\nObserving...")
    observation = self._observe(query, thoughts, action)
    return observation
  def run(self, query, verbose = False):
    if verbose:
      return self._run_verbose(query)
    else:
      return self._run_normal(query)