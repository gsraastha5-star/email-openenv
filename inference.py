import os
import random

from openai import OpenAI

from env import EmailEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

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
        content = response.choices[0].message.content or ""
        label = content.strip().lower()
        return label if label in VALID_ACTIONS else random.choice(VALID_ACTIONS)
    except Exception:
        return random.choice(VALID_ACTIONS)


def run_task(env: EmailEnv, client: OpenAI, task: str) -> float:
    env.reset(task)
    done = False
    step_num = 0

    print(f"[START] task={task}")

    while not done:
        step_num += 1
        obs = env._get_observation()
        if obs is None:
            break

        action = get_action(client, obs)
        result = env.step(action)
        done = result.done

        reward_value = result.reward.value if result.reward else 0.0
        reason = result.reward.reason if result.reward else ""

        print(
            f"[STEP] task={task} step={step_num} "
            f"subject={obs.subject!r} sender={obs.sender!r} "
            f"action={action} reward={reward_value:.2f} done={done} reason={reason!r}"
        )

    score = env.get_grader_score()
    print(
        f"[END] task={task} score={score:.4f} "
        f"accuracy={score * 100:.1f}% total_reward={env.total_reward:.2f}"
    )
    return score


if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    env = EmailEnv()
    scores = {}

    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(env, client, task)

    print(
        "[END] summary "
        + " ".join(f"{task}={score:.4f}" for task, score in scores.items())
    )
