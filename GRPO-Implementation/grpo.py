import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POLICY_MODEL_NAME = "gpt2"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large"

NUM_SAMPLES = 8
MAX_NEW_TOKENS = 64
LR = 1e-5

policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
policy_tokenizer.pad_token = policy_tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(
    POLICY_MODEL_NAME
).to(DEVICE)

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_NAME
).to(DEVICE)
reward_model.eval()

for p in reward_model.parameters():
    p.requires_grad = False

def generate_group(prompt):
    inputs = policy_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = policy_model.generate(
        **inputs,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        num_return_sequences=NUM_SAMPLES,
        max_new_tokens=MAX_NEW_TOKENS,
        return_dict_in_generate=True
    )

    sequences = outputs.sequences
    texts = policy_tokenizer.batch_decode(
        sequences, skip_special_tokens=True
    )

    return sequences, texts

def score_responses(prompt, responses):
    inputs = [
        prompt + "\n\n" + r for r in responses
    ]

    enc = reward_tokenizer(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        scores = reward_model(**enc).logits.squeeze(-1)

    return scores

def grpo_step(prompt):
    policy_model.train()

    sequences, texts = generate_group(prompt)
    rewards = score_responses(prompt, texts)

    advantages = rewards - rewards.mean()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = []

    for seq in sequences:
        input_ids = seq[:-1]
        target_ids = seq[1:]

        logits = policy_model(
            input_ids.unsqueeze(0)
        ).logits

        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(
            2, target_ids.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)

        log_probs.append(token_logp.sum())

    log_probs = torch.stack(log_probs)

    loss = -(log_probs * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), texts, rewards.tolist()

prompt = "Explain why regular exercise is important."

for step in range(50):
    loss, texts, rewards = grpo_step(prompt)

    print(f"\nStep {step} | Loss: {loss:.4f}")
    ranked = sorted(zip(rewards, texts), reverse=True)

    for r, t in ranked:
        print(f"[{r:+.2f}] {t[:100]}...")
