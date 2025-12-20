import mesa
import mesa_llm   # for context (Mesa-LLM style project)
import ollama     # local Llama 3


# ----------------------------
# 1. Agent definition
# ----------------------------
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


# ----------------------------
# 2. Model definition
# ----------------------------
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


# ----------------------------
# 3. Run the model
# ----------------------------
if __name__ == "__main__":
    print("Starting Mesa (3.x) + local Llama 3 (Ollama) simulation...")
    model = LanguageModel(n_agents=5)
    model.step()  # all agents call the LLM once
