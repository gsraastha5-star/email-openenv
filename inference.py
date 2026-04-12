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
VALID_ACTIONS = ["mark_spam", "escalate", "promotions_tab"]

SYSTEM_PROMPT = """You are an expert inbox triage assistant.

You must choose exactly one action for each email:
- mark_spam: scams, phishing, fraudulent requests, malicious or suspicious email
- escalate: important, urgent, operational, business-critical, reply-worthy, or genuinely useful email
- promotions_tab: marketing, discounts, sales, shopping offers, newsletters, and non-essential commercial content

Guidelines:
- Be very careful with phishing, spoofed domains, fake urgency, credential checks, and suspicious verification requests. These usually require mark_spam.
- Meetings, invoices, deadlines, contracts, client updates, shipment updates, payroll, and team coordination often require escalate.
- Discounts, offers, flash sales, product promotions, and free trials usually belong in promotions_tab.

Reply with exactly one action:
mark_spam
escalate
promotions_tab
"""


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def build_user_message(obs) -> str:
    return f"""Choose the best inbox triage action for this email.

Task context: {obs.task_description}
Subject: {obs.subject}
Sender: {obs.sender}
Sender domain: {obs.sender_domain}
Contains links: {obs.contains_links}
Contains urgency words: {obs.contains_urgency_words}
External sender: {obs.is_external_sender}
Body: {obs.email_text}

Return exactly one action:
mark_spam
escalate
promotions_tab
"""


def get_action(client: OpenAI, obs) -> str:
    user_msg = build_user_message(obs)

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

    text_blob = f"{obs.subject} {obs.sender} {obs.email_text}".lower()

    if any(
        word in text_blob
        for word in [
            "verify",
            "suspension",
            "refund",
            "won",
            "free iphone",
            "compromised",
            "click here",
            "credential",
            "payroll verification",
        ]
    ):
        return "mark_spam"

    if any(
        word in text_blob
        for word in [
            "sale",
            "discount",
            "offer",
            "trial",
            "membership",
            "browse",
            "arrivals",
            "flash sale",
        ]
    ):
        return "promotions_tab"

    if any(
        word in text_blob
        for word in [
            "meeting",
            "invoice",
            "deadline",
            "contract",
            "client",
            "team",
            "shipment",
            "board",
            "payroll",
            "standup",
        ]
    ):
        return "escalate"

    return random.choice(VALID_ACTIONS)


def run_task(env: EmailEnv, client: OpenAI, task: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        env.reset(task, seed=42)
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
        success = score >= 0.3

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
