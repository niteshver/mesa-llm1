# Creating Your First mesa-llm Model
[Mesa LLM](https://github.com/mesa/mesa-llm) is a set of tools that combines Large Language Models (LLMs) with [Agent-Based Modeling](https://en.wikipedia.org/wiki/Agent-based_model) (ABM) using the Mesa framework. It allows developers and researchers to build simulations where agents powered by LLMs can communicate, reason, remember, and make decisions inside realistic environments such as markets, organizations, or societies.

This approach is especially useful for studying how complex and emergent behaviors arise from simple agent interactions. Mesa LLM also helps in designing more human-like, language-driven agents and provides tools (such as MESA for Inference) to analyze and evaluate LLM performance. Overall, it enables the creation of smarter, more interactive AI systems for applications ranging from economic simulations to advanced code review and analysis.

## Model Discription
In Mesa LLm, the Agents is that reason using language inside a Mesa simulation loop.Each agent is capable of processing a textual prompt and generating a natural-language response during its reasoning step.

## What the model does
* The model initializes a small number of agents. Each agent represents an entity capable of reasoning using language.
* During each simulation step, the model defines a simple text input (for example, a question) that will be given to all agents.
* The agents’ responses are printed or recorded.

## Tutorial Setup
Create and activate a virtual environment. Python version 3.12 or higher is required.

## Install Mesa LLM & required package

Install Mesa LLM

```bash
pip install -U mesa-llm
```

Mesa-LLM pre-releases can be installed with:
```bash
pip install -U --pre mesa-llm
```

You can also use pip to install the GitHub version:
```bash
pip install -U -e git+https://github.com/mesa/mesa-llm.git#egg=mesa-llm
```
Install Ollama & llama3
```bash
pip install ollama
ollama run llama3
```
You can also install [Ollama](https://ollama.com/) from official website


Or any other (development) branch on this repo or your own fork:
```bash
pip install -U -e git+https://github.com/YOUR_FORK/mesa-llm@YOUR_BRANCH#egg=mesa-llm
```


### Mesa-LLM supports the following LLM models:
* OpenAI
* Anthropic
* xAI
* Huggingface
* Ollama
* OpenRouter
* NovitaAI
* Gemini

## Building the Model
After Mesa LLm is installed a model can be built.
This tutorial is written in [Jupyter](https://jupyter.org/) to facilitate the explanation portions.

Start Jupyter form the command line:
 
 Jupyter lab

Create a new notebook named example.ipynb or whatever you want.

## Important Dependencies
This includes importing of dependencies needed for the tutorial.
```bash
import mesa_llm
```

## Creating the Agent
We begin by defining a minimal agent that uses a language model to reason about its behavior at each simulation step. The agent represents an individual participant in the model that receives a textual prompt and produces a natural-language response describing its intended action.

Although the agent relies on language-based reasoning, it remains a standard Mesa agent. It inherits from mesa.Agent, follows Mesa’s normal execution loop, and is automatically assigned a unique_id by the model. Mesa continues to handle scheduling and activation without any modification.

The main difference lies in how decisions are made during a step. Instead of using hard-coded rules, the agent builds a short natural-language prompt describing its role in the simulation and sends it to a language model for reasoning. In this tutorial, we use Ollama with a local Llama 3 model as the language backend.

The generated response is treated as the agent’s reasoning output and printed directly, allowing us to observe how different agents interpret the same simulation context.
The LanguageAgent class is created with the following code:
```bash
class LanguageAgent(mesa.Agent):
    """
    A simple LLM-powered agent in Mesa-LLM style.
    """

    def __init__(self, model):
        # In Mesa 3, unique_id is assigned automatically by the model
        super().__init__(model=model)

    def step(self):
        """
        One step of the agent:
        - Build a prompt with its unique_id.
        - Ask the local Llama 3 model (via Ollama).
        - Print the response.
        """
        prompt = (
            f"You are agent {self.unique_id} in a simple market simulation. "
            f"Other agents are trading and negotiating. "
            f"Describe your next action in one short sentence."
        )

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response["message"]["content"].strip()
        print(f"Agent {self.unique_id}: {text}")
```

## Create the Model
After defining the agent, we create the model that manages the simulation.
In Mesa, the model acts as a container for all agents and is responsible for their creation, scheduling, and execution over time.

In this tutorial, the model also stores the language model client. This allows all agents to share the same LLM instance instead of creating one per agent, which is both simpler and more efficient.

When a LanguageModel is initialized, the number of agents is specified. The model then creates a scheduler, instantiates each agent with a unique identifier, and adds them to the scheduler. During each model step, the scheduler activates the agents one by one, triggering their language-based reasoning.

This structure follows the standard Mesa workflow. The model now holds an LLM client, which agents use during their step() method.

The LanguageModel class is created with the following code:
```bash 
class LanguageModel(mesa.Model):
    """
    Mesa model with LLM-powered agents (Mesa 3 style).
    """

    def __init__(self, n_agents: int = 5, seed: int | None = None):
        # Mesa 3 requires calling super().__init__
        super().__init__(seed=seed)

        # Create agents; Mesa 3 automatically tracks them in model.agents
        for _ in range(n_agents):
            LanguageAgent(model=self)

    def step(self):
        """
        Advance the model by one step.

        Mesa 3: use AgentSet API for activation.
        shuffle_do("step") = random order, all agents call step().
        """
        self.agents.shuffle_do("step")  # random activation of all agents
```        


## Running the Model

The model is initialized with a fixed number of agents (example = 5), each identified by a unique unique_id.
When the model step is executed, the scheduler activates all agents once. During their activation, each agent performs its step() method, where it receives a text prompt and generates a language-based response using the shared LLM client.
```bash
if __name__ == "__main__":
    print("Starting Mesa-LLM with Ollama...")
    model = LanguageModel(5)
    model.step()  # One step = all 5 agents call the LLM once
```











