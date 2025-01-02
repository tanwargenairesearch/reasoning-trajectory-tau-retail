# Generating Reasoning Trajectories in a Simulated Customer Support Environment using TAU-Bench and Smolagents

## Setup

1. Clone this repository:

```bash
git clone https://github.com/tanwargenairesearch/reasoning-trajectory-tau-retail && cd ./reasoning-trajectory-tau-retail
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```

3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

## Run

```bash
python src/run.py --model gemini-2.0-flash-exp --model-provider google --user-model gpt-4o --user-model-provider openai --user-strategy llm
```

# Rationale

Tired of chatbots that can't handle complex requests? Let's build an AI that can strategize and solve problems just like a human support agent – but faster and at scale.

Imagine this: A customer wants to change their order, update payment info, AND adjust the delivery address. Most chatbots would crash and burn 🔥.

By combining [Tau-bench](https://github.com/sierra-research/tau-bench) and [Smolagent](https://github.com/huggingface/smolagents), we can generate reasoning trajectories – step-by-step solutions that factor in customer needs, policies, and product knowledge.

Think of it like a real-time strategy game where the AI agent needs to make the right moves to win (aka, a happy customer!).

Here's how it works:

- LLMs to simulate both the customer AND the agent, creating dynamic, realistic conversations.
- Tau-bench provides the playing field (simulated scenarios + grounding facts).
- Smolagent helps the AI agent reason through the problem and find the best solution.

- The result? A dataset that can train the next generation of super-smart support chatbots.
