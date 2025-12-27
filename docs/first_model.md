# Creating Your First mesa-llm Model

## Tutorial Description
This tutorial introduces mesa-llm by walking through the construction of a simple language-driven agent model built on top of Mesa. While Mesa traditionally relies on rule-based agent behavior, mesa-llm enables agents to reason and decide using natural language through integration with large language models.

In this tutorial, we build a minimal model in which each agent uses a language model to describe its next action at every simulation step. The goal is not to create a complex simulation, but to clearly demonstrate how language-based reasoning can be embedded into Mesa’s existing execution workflow.

By the end of this tutorial, you will understand how mesa-llm fits into the Mesa framework, how language reasoning replaces traditional decision logic, and how different language model backends (such as Ollama) can be integrated without modifying the overall simulation structure.

## About mesa-llm

[Mesa LLM](https://github.com/mesa/mesa-llm) is a set of tools that integrates Large Language Models (LLMs) with [Agent-based modeling](https://en.wikipedia.org/wiki/Agent-based_model) using the Mesa framework. It enables simulations in which agents use natural language to reason, communicate, and make decisions while still operating within Mesa’s standard modeling and scheduling workflow.

This approach is particularly useful for exploring how complex or emergent behavior can arise from interactions between language-driven agents. By combining Mesa’s structured simulation environment with LLM-based reasoning, Mesa LLM allows researchers and developers to experiment with more flexible, human-like agent behavior in settings such as markets, organizations, or social systems.

Overall, Mesa LLM focuses on providing a practical bridge between traditional agent-based models and modern language models, making it easier to study and prototype simulations where decision-making is driven by natural language rather than fixed rules.

## Model Description
The model consists of a fixed number of agents, each representing an independent participant in a simple simulated environment. At each step of the simulation, every agent is activated once and asked to reason about its next action using natural language.

Instead of relying on hard-coded rules or conditional logic, agents generate their behavior by responding to a textual prompt that describes their role and context within the model. This response is produced by a language model and treated as the agent’s reasoning output for that step.

Importantly, the rest of the simulation remains unchanged from a standard Mesa model. Mesa continues to manage agent creation, scheduling, and execution order, while mesa-llm is responsible only for how agents reason and generate decisions. This separation allows language-based reasoning to be added to existing Mesa models with minimal changes to their overall structure.

## What the model does
* The model initializes a small number of agents. Each agent represents an entity capable of reasoning using language.
* During each simulation step, each agent constructs its own text prompt describing the current situation and uses it to reason about its next action.
* The agents’ responses are printed or recorded.

## Tutorial Setup
Create and activate a virtual environment. Python version 3.12 or higher is required.

## Install mesa-llm and required package

Install mesa-llm

```bash
pip install -U mesa-llm
```

Mesa-llm pre-releases can be installed with:
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
The command downloads and runs the Llama 3 model locally.

You can also install [Ollama](https://ollama.com/) from official website


Or any other (development) branch on this repo or your own fork:
```bash
pip install -U -e git+https://github.com/YOUR_FORK/mesa-llm@YOUR_BRANCH#egg=mesa-llm
```


### Mesa-llm supports the following LLM models:
* OpenAI
* Anthropic
* xAI
* Huggingface
* Ollama
* OpenRouter
* NovitaAI
* Gemini

## Building the Model
After Mesa-llm is installed a model can be built.
This tutorial can be followed in a regular Python script or in a [Jupyter](https://jupyter.org/) notebook.


Start Jupyter from the command line:
 ```bash
 jupyter lab
 ```

Create a new notebook named example.ipynb or whatever you want.

## Important Dependencies
This includes importing of dependencies needed for the tutorial.
```python
import mesa_llm
import mesa
import ollama
```

## Creating the Agent
In this example, the agent is prompted to make its reasoning explicit before deciding on an action. Instead of directly producing a decision, the agent first explains how it interprets the current situation and then states what it intends to do. Making this reasoning step visible helps illustrate how language-based cognition can be integrated into Mesa’s execution loop, and allows us to observe not only agent actions but also the thought process that led to them.

We begin by defining a minimal agent that uses a language model to reason about its behavior at each simulation step. The agent represents an individual participant in the model that receives a textual prompt and produces a natural-language response describing its intended action.

Although the agent relies on language-based reasoning, it remains a standard Mesa agent. It inherits from mesa.Agent, follows Mesa’s normal execution loop, and is automatically assigned a unique_id by the model. Mesa continues to handle scheduling and activation without any modification.

The main difference lies in how decisions are made during a step. Instead of using hard-coded rules, the agent builds a short natural-language prompt describing its role in the simulation and sends it to a language model for reasoning. In this tutorial, we use Ollama with a local Llama 3 model as the language backend.

The generated response is treated as the agent’s reasoning output and printed directly, allowing us to observe how different agents interpret the same simulation context.
The LanguageAgent class is created with the following code:
```python
class LanguageAgent(mesa.Agent):
    """
    A simple LLM-powered agent in Mesa-llm style.
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
            f"You are agent {self.unique_id} in a simple market simulation.\n"
            f"Other agents are trading and negotiating.\n\n"
            f"Please respond in the following format:\n"
            f"Reasoning:\n"
            f"<one short paragraph>\n\n"
            f"Action:\n"
            f"<one short sentence>"
        )

        response = ollama.chat(
            model="llama3",                                # use llama3 via ollama
            messages=[{"role": "user", "content": prompt}],
        )

        text = response["message"]["content"].strip()      
        print("\n" + "=" * 60)
        print(f"Agent {self.unique_id}")
        print(text)
        print("=" * 60 + "\n")
```

## Create the Model
After defining the agent, we create the model that manages the simulation.
In Mesa, the model acts as a container for all agents and is responsible for their creation, scheduling, and execution over time.

When a LanguageModel is initialized, the number of agents is specified. The model instantiates each agent, which is automatically tracked by Mesa’s internal AgentSet and assigned a unique identifier.
During each model step, all agents are activated once in a randomized order using Mesa’s AgentSet API and  triggering their language-based reasoning. This structure follows the standard Mesa workflow. 

The LanguageModel class is created with the following code:
```python
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

The model is initialized with a fixed number of agents (example = 5), each identified by a unique_id.
When the model step is executed, the scheduler activates all agents once. 
During their activation, each agent performs its step() method, where it receives a text prompt and generates a language-based response using the configured LLM backend,
```python
if __name__ == "__main__":
    print("Starting Mesa-llm with Ollama...")
    model = LanguageModel(5)
    model.step()  # One step = all 5 agents call the LLM once
```

Note: The following output is an example of one possible run. Language model responses may vary, and not all agents will strictly follow the requested format.

```bash
Starting mesa-llm with Ollama...

============================================================
Agent 5
I'm ready to play. Here's my response:

Reasoning:
As I review the current prices, I notice that the demand for goods is relatively high, which could lead to an increase in prices. Additionally, some of my fellow agents seem eager to buy, which may drive up the cost.

Action:
I decide to hold back on selling and observe the market dynamics a bit longer before making any moves.
============================================================

============================================================
Agent 4
Let's start!

Reasoning: I've been observing the market trends and noticed that Agent 3 has been consistently buying up supply of Goods X. This might be a sign of increased demand, which could drive prices up in the long run.

Action: I'll offer to sell my current stockpile of Goods Y to Agent 2 at a slightly higher price than usual to capitalize on potential future demand and profit from any price increases.
============================================================

============================================================
Agent 3
Let's get started. Here is my response:

Reasoning:
I've been observing the market for a while, and I think there might be an opportunity to profit from trading with agent 1, who seems to be getting too aggressive in their pricing.

Action:
I will make an offer of 10 units for 50 coins to agent 1.
============================================================
```

## Next Steps

- Try different prompts to change agent behavior.
- Experiment with different numbers of agents.
- Replace Ollama with another supported LLM backend.
- Extend the model to allow agents to communicate with each other.














