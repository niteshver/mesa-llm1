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

## Install Mesa LLM

Install Mesa LLM

```bash
pip install -U mesa-llm

Mesa-LLM pre-releases can be installed with:
```bash
pip install -U --pre mesa-llm

You can also use pip to install the GitHub version:
```bash
pip install -U -e git+https://github.com/mesa/mesa-llm.git#egg=mesa-llm



Or any other (development) branch on this repo or your own fork:
```bash
pip install -U -e git+https://github.com/YOUR_FORK/mesa-llm@YOUR_BRANCH#egg=mesa-llm


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

## Creating the Agent
We begin with a minimal agent definition and gradually extend its behavior throughout the tutorial.

Although the agent uses language-based reasoning, it is still a standard Mesa agent. It follows Mesa’s normal execution model and relies on the same unique_id mechanism for identification, model access, and scheduling.

During each model step, the agent receives a text prompt and produces a language-based response. This replaces traditional rule-based decision logic with language-driven reasoning, while keeping the rest of the Mesa workflow unchanged.
The LanguageAgent class is created with the following code:
```bash
class LanguageAgent(mesa.Agent):
    """
    Mesa-LLM agent that performs language-based reasoning.

    This agent is a regular Mesa agent and inherits all standard
    Mesa functionality, including unique_id handling and access
    to the parent model.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


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
    Mesa model that manages LLM-powered agents.
    """
    def __init__(self, n_agents=5):
        super().__init__()

        # Shared LLM client (Ollama)
        self.llm = llm

        # Scheduler controls agent activation order
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents and add them to the scheduler
        for i in range(n_agents):
            agent = LanguageAgent(i, self)
            self.schedule.add(agent)

    def step(self):
        # Activate all agents (each agent calls the LLM in its step)
        self.schedule.step()












