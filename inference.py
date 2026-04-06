import os
import random

from openai import OpenAI

from env import EmailEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

VALID_ACTIONS = ["spam", "important", "promotion"]

SYSTEM_PROMPT = """You are an email classifier.
Given an email, classify it as exactly one of: spam, important, promotion.
Reply with only one word: spam, important, or promotion."""


def get_action(client: OpenAI, obs) -> str:
    user_msg = f"""Subject: {obs.subject}
From: {obs.sender}
Email: {obs.email_text}

Classify this email as exactly one of: spam, important, promotion.
Reply with one word only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=10,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in VALID_ACTIONS else random.choice(VALID_ACTIONS)
    except Exception as exc:
        print(f"[LLM error: {exc}] using random fallback")
        return random.choice(VALID_ACTIONS)


def run_task(env: EmailEnv, client: OpenAI, task: str) -> float:
    print(f"\n{'=' * 50}")
    print(f"TASK: {task.upper()}")
    print(f"{'=' * 50}")

    env.reset(task)
    done = False
    step_num = 0

    while not done:
        step_num += 1
        obs = env._get_observation()
        if obs is None:
            break

        action = get_action(client, obs)

        print(f"\nStep {step_num}:")
        print(f"  Subject: {obs.subject}")
        print(f"  Sender : {obs.sender}")
        print(f"  Email  : {obs.email_text}")
        print(f"  Chose  : {action}")

        result = env.step(action)
        print(f"  Reward : {result.reward.value:+.1f} ({result.reward.reason})")
        done = result.done

    score = env.get_grader_score()
    print(f"\nFinal Score ({task}): {score:.2f} ({score * 100:.1f}% accuracy)")
    print(f"Total Reward: {env.total_reward:.2f}")
    return score


if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    env = EmailEnv()
    scores = {}

    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(env, client, task)

    print(f"\n{'=' * 50}")
    print("BASELINE SUMMARY")
    print(f"{'=' * 50}")
    for task, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {task:<8}: {score:.2f}  {bar}")
