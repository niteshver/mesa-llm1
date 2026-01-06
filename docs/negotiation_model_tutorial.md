# Negotiation Model Tutorial (mesa-llm)

## About This Tutorial

This tutorial is inspired by the negotiation example provided in the mesa-llm repository:
``` bash
https://github.com/mesa/mesa-llm/tree/main/examples/negotiation
```
The goal here is **not to replicate** the original example, but to present a **simplified and tutorial-friendly version** that focuses on reasoning structure and agent interaction.

Key differences from the original example include:
- A reduced number of agents
- No enforced negotiation protocol
- No spatial environment or grid
- Emphasis on understanding ReAct reasoning output

This makes the example easier to follow for new users while still demonstrating core mesa-llm concepts.

## Model Description

This model simulates a basic negotiation scenario involving:
- One seller agent with a minimum acceptable price
- Two buyer agents, each with a different budget

### Each buyer:

- Has a different budget
- Reasons independently using ReActReasoning
- Buyers do not coordinate with each other

### Seller Agent

- Inherits from `LLMAgent`
- Uses `ReActReasoning` to reason about negotiation decisions
- Considers its minimum acceptable price and the current simulation step


## Tutorial Setup
Ensure you are using Python 3.12 or later.

## Install mesa-llm and required packages

Install mesa-llm

```bash
pip install -U mesa-llm
```

## Why This Model Is Non-Spatial

Negotiation is a conceptual process rather than a spatial one.
Buyers and sellers negotiate based on preferences, constraints, and reasoning—not physical position.
To keep this tutorial simple and focused, no grid or physical environment is introduced.
Adding a grid would increase complexity without improving the clarity of the negotiation logic.


## Model Execution
At each model step:
- The model advances one step
- All agents are activated using shuffle_do("step")
Each agent generates a reasoning plan using ReActReasoning and applies it.
The console output displays the internal reasoning traces.

## Creating the Seller class (Agent)
Using the previously imported dependencies, we define the agent class:
The Seller agent inherits from LLMAgent and uses ReActReasoning to decide how to respond during negotiation.
It reasons about the current simulation step and its minimum acceptable price.

``` python
# Import Dependencies
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.memory.st_lt_memory import STLTMemory

# ---------------- SELLER ----------------
class Seller(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",
        )
    def step(self):
        observation = {
            "step": self.model.steps,
            "min_price": self.internal_state["min_price"]
        }
        prompt = """
        You are a seller negotiating the price of a single item.
        You want to sell as high as possible, but not below your minimum price.
        Reason about your decision.
        """
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation
        )
        self.apply_plan(plan)
```

## Creating the Buyer Class
### Each buyer:
- Has a different budget
- Reasons independently using ReActReasoning
- Buyers do not coordinate with each other

``` python

# ---------------- BUYER ----------------
class Buyer(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",
        )
    def step(self):
        observation = {
            "step": self.model.steps,
            "budget": self.internal_state["budget"]
        }
        prompt = """
        You are a buyer negotiating to purchase a single item.
        You have a fixed budget and want the lowest possible price.
        Reason about your decision.
        """
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation
        )
        self.apply_plan(plan)
```
## Creating Negotiation Model
The NegotiationModel sets up and runs the negotiation between buyers and a seller.
It creates the agents, assigns their roles, and controls when each agent acts.

### How it works:
- Inherits from Mesa’s Model class.
- Creates one seller agent with a minimum acceptable price.
- Creates two buyer agents with different budgets.
- Assigns ReActReasoning to all agents so they can reason using an LLM.
- Uses shuffle_do("step") to let all agents act once per model step.
- Prints the current step number to track simulation progress.
- Repeats this process for a fixed number of steps in the main loop.
``` python
# ---------------- MODEL ----------------
class NegotiationModel(Model):
    def __init__(self, llm_model="ollama/llama3", seed=None):
        super().__init__(seed=seed)

        Seller.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt="You are a seller.",
            internal_state={"min_price": 60},
        )
        Buyer.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt="You are a buyer.",
            internal_state={"budget": 100},
        )
        Buyer.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt="You are a buyer.",
            internal_state={"budget": 70},
        )
    def step(self):
        print(f"\n--- Model step {self.steps} ---")
        self.agents.shuffle_do("step")
```
## Running the Model
- The model runs for a few steps using a loop.
- In each step, all agents think and act once.
- The reasoning output is printed to the console.
``` python
# ---------------- RUN ----------------
if __name__ == "__main__":
    model = NegotiationModel()
    for _ in range(3):
        model.step()
```

## Understanding the Output
Below is an example of the reasoning output produced by ReActReasoning:
``` bash
[Plan]                                                                                    │
│    └── reasoning : As I have just started, my short-term and long-term memories are       │
│ empty. My current observation shows that this is the first step with a budget of 100.     │
│ Given these circumstances, I decide to move one step forward to explore the environment   │
│ and see what it has to offer.                                                             │
│    └── action : move_one_step 
```

## What’s Next

This tutorial intentionally keeps the negotiation logic minimal.
Possible extensions include:
- Adding a shared negotiation state (e.g., tracking the current offered price)
- Enforcing turn-taking between buyers and the seller
- Restricting ReAct actions to task-specific decisions
- Introducing a spatial environment (grid) to simulate marketplace-style scenarios
- Extending the model to support multiple items or multiple sellers
