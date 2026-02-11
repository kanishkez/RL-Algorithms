# RL Algorithms

This repository contains from-scratch implementations of reinforcement learning algorithms using PyTorch and Gymnasium, with a focus on understanding algorithms deeply rather than using high-level libraries.

Currently included:
- **GRPO (Group Relative Policy Optimization)** — Language model fine-tuning with RLHF
- **PPO (Proximal Policy Optimization)** — LunarLander-v3 with Actor-Critic and GAE
- **REINFORCE Policy Gradient** — CartPole-v1
- **Actor–Critic with Self-Play** — Custom Tic-Tac-Toe environment

---

## GRPO — Language Model Fine-Tuning with RLHF

A minimal implementation of **Group Relative Policy Optimization (GRPO)** for fine-tuning language models using reinforcement learning from human feedback (RLHF).

### Overview

This script demonstrates how to:
- Generate multiple responses from a language model
- Score them using a reward model trained on human preferences
- Update the policy model to favor higher-scoring responses

GRPO uses the group average reward as a baseline, making it simpler and more stable than traditional PPO implementations.

### Requirements

```bash
pip install torch transformers accelerate
```

**Hardware:**
- GPU recommended (CUDA-compatible)
- Minimum 8GB VRAM for default models
- Works on CPU but will be significantly slower

### Algorithm

GRPO updates the policy in the direction of higher-than-average rewards:

1. **Generate K responses** from the current policy
2. **Score each response** using the reward model
3. **Calculate advantages**: `advantage = reward - mean(rewards)`
4. **Normalize advantages** for stability
5. **Update policy**: $\theta \leftarrow \theta - \alpha \nabla_\theta \sum_k \log \pi_\theta(a_k \mid s_k) \cdot A_k$

### Quick Start

```bash
python grpo_training.py
```

The script will train GPT-2 for 50 steps on the prompt: *"Explain why regular exercise is important."*

### Configuration

All configuration is at the top of the script:

### Device Selection

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Changing the Policy Model

The policy model is what you're training. You can use any causal language model from Hugging Face:

```python
POLICY_MODEL_NAME = "gpt2"  # Default

# Other options:
# POLICY_MODEL_NAME = "gpt2-medium"
# POLICY_MODEL_NAME = "gpt2-large"
# POLICY_MODEL_NAME = "EleutherAI/gpt-neo-125M"
# POLICY_MODEL_NAME = "EleutherAI/pythia-410m"
# POLICY_MODEL_NAME = "facebook/opt-350m"
```

**Note:** Larger models require more VRAM and train slower.

### Changing the Reward Model

The reward model scores how good responses are. You can use any sequence classification model:

```python
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large"  # Default

# Other options:
# REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"  # Smaller, faster
# REWARD_MODEL_NAME = "reciprocate/pairrm-hf"  # Alternative reward model
```

**Important:** Different reward models may have different input formats. The current implementation assumes a simple `[prompt + response]` format.

### Training Hyperparameters

```python
NUM_SAMPLES = 8           # Number of responses to generate per step
MAX_NEW_TOKENS = 64       # Maximum length of generated responses
LR = 1e-5                 # Learning rate for policy model optimizer
```

**Tuning tips:**
- `NUM_SAMPLES`: Higher = more stable but slower. Try 4-16.
- `MAX_NEW_TOKENS`: Longer responses = more tokens = slower training
- `LR`: Too high causes instability, too low slows learning. Try 5e-6 to 5e-5.

### Generation Parameters

In the `generate_group()` function:

```python
outputs = policy_model.generate(
    **inputs,
    do_sample=True,
    temperature=1.0,      # Higher = more random, lower = more focused
    top_p=0.95,           # Nucleus sampling threshold
    num_return_sequences=NUM_SAMPLES,
    max_new_tokens=MAX_NEW_TOKENS,
)
```

### Customizing the Training

#### Change the Training Prompt

In the training loop at the bottom:

```python
prompt = "Explain why regular exercise is important."

# Change to anything:
# prompt = "Write a short story about a robot."
# prompt = "What are the benefits of meditation?"
```

#### Multiple Prompts

For diverse training, modify the loop:

```python
prompts = [
    "Explain why regular exercise is important.",
    "Describe the water cycle.",
    "What is machine learning?"
]

for step in range(50):
    prompt = prompts[step % len(prompts)]
    loss, texts, rewards = grpo_step(prompt)
    # ...
```

#### Adjust Number of Training Steps

```python
for step in range(50):  # Change 50 to any number
```

### Files
- Training: `grpo_training.py`
- Saved model weights: `trained_model/` (after training)

### Results
- Model learns to generate higher-scoring responses
- Responses improve in coherence and alignment with reward model preferences
- Training typically shows decreasing loss and increasing mean rewards

---

## LunarLander-v3 — Proximal Policy Optimization (PPO)

This project solves LunarLander-v3 using Proximal Policy Optimization (PPO) with an Actor–Critic architecture and Generalized Advantage Estimation (GAE).

This implementation is written fully from scratch using PyTorch.

### Algorithm

PPO constrains policy updates to prevent destabilizing large changes:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

**Key features:**
- Clipped surrogate objective prevents large policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Value function updates to estimate state values

### Files
- Training: `lunarlander_ppo.py`
- Evaluation (with rendering): `lunar_eval.py`
- Saved model weights: `ppo_lunarlander_final.pt`

### Train
```bash
python lunarlander_ppo.py
```

### Evaluate (with visualization)
```bash
python lunar_eval.py
```

### Results
- Consistently achieves episode rewards ≥ 200
- Stable landings in most evaluation episodes
- Occasional failures expected due to stochastic dynamics

---

## CartPole-v1 — REINFORCE Policy Gradient

This project demonstrates solving CartPole-v1 using the REINFORCE (Monte-Carlo Policy Gradient) algorithm.

### Highlights
- Pure policy-gradient method (no value function)
- Episode-level return computation
- Trained until the environment is considered solved

### Algorithm

REINFORCE updates the policy in the direction of higher expected return:

$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t$$

### Run
```bash
python train_cartpole.py
```

### Inference
```bash
python cartpoleinference.py
```

### Results
- Typically solves CartPole-v1 within 500–1500 episodes
- Average reward ≥ 195
- Trained model weights (.pth) are included in the repository

---

## Tic-Tac-Toe — Actor–Critic with Self-Play

This project implements a custom Tic-Tac-Toe environment and trains an Actor–Critic agent using self-play against a frozen opponent.

### What's special here
- Custom environment written from scratch (`tictactoe_env.py`)
- Actor–Critic architecture:
  - **Actor**: learns the policy
  - **Critic**: learns the value function
- Self-play using an older frozen version of the agent
- Human-vs-agent evaluation

### Files
- Environment: `tictactoe_env.py`
- Training: `tictactoe_actorcritic.py`
- Evaluation (human vs agent): `tictactoe_eval.py`
- Saved weights: actor and critic .pth files included

### Train
```bash
python tictactoe_actorcritic.py
```

### Evaluate (Human vs Agent)
```bash
python tictactoe_eval.py
```

The starting player is randomized so the agent must play both first and second.

### Outcome
- Agent learns optimal or near-optimal Tic-Tac-Toe play
- Avoids illegal moves
- Consistently beats random and weak opponents
- Demonstrates stable learning via self-play

---

## Requirements

- Python ≥ 3.8
- PyTorch
- Gymnasium
- NumPy
- Transformers (for GRPO only)
- Accelerate (for GRPO only)

## Goal of This Repository

This repo is meant to be a learning-first RL codebase, showing:
- How RL algorithms actually work internally
- How to write custom environments
- How to train, evaluate, and debug agents without shortcuts
- Progression from simple policy gradients to advanced methods like PPO and GRPO

More algorithms and environments will be added over time.

## License

MIT License - feel free to use and modify for your projects.
