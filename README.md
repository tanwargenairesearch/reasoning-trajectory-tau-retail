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
