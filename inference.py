import os
import random

from openai import OpenAI

from env import EmailEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-openenv"
VALID_ACTIONS = ["spam", "important", "promotion"]

SYSTEM_PROMPT = """You are an email classifier.
Given an email, classify it as exactly one of: spam, important, promotion.
Reply with only one word: spam, important, or promotion."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
            temperature=0,
            max_tokens=10,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip().lower()
        if content in VALID_ACTIONS:
            return content
    except Exception:
        pass

    return random.choice(VALID_ACTIONS)


def run_task(env: EmailEnv, client: OpenAI, task: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        env.reset(task)
        done = False

        while not done:
            obs = env._get_observation()
            if obs is None:
                break

            action = get_action(client, obs)
            result = env.step(action)

            reward = result.reward.value if result.reward else 0.0
            done = result.done
            error = None

            steps_taken += 1
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

        score = env.get_grader_score()
        success = score > 0.0

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    for task_name in TASKS:
        env = EmailEnv()
        run_task(env, client, task_name)
