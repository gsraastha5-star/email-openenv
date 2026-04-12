from typing import Optional
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Observation(BaseModel):
    email_text: str
    sender: str
    sender_domain: str
    subject: str
    contains_links: bool
    contains_urgency_words: bool
    is_external_sender: bool
    step: int
    total_emails: int
    task: str
    task_description: str


class Action(BaseModel):
    label: str


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None


class Reward(BaseModel):
    value: float
    reason: str


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: dict


TASK_METADATA = {
    "easy": {
        "description": "Personal inbox triage with obvious spam, promotions, and important messages.",
    },
    "medium": {
        "description": "Workplace inbox triage with invoices, deadlines, renewals, and mixed-intent emails.",
    },
    "hard": {
        "description": "Security-sensitive inbox triage with phishing-like, ambiguous, and high-stakes business emails.",
    },
}


DATASETS = {
    "easy": [
        {
            "text": "Win a free iPhone now! Click here!",
            "sender": "promo@scam.com",
            "subject": "You won!!",
            "label": "mark_spam",
        },
        {
            "text": "Team meeting at 5 PM today.",
            "sender": "boss@company.com",
            "subject": "Meeting reminder",
            "label": "escalate",
        },
        {
            "text": "50% off on all shoes this weekend. Shop now.",
            "sender": "shop@store.com",
            "subject": "Weekend Sale",
            "label": "promotions_tab",
        },
        {
            "text": "Your account has been compromised. Verify immediately here.",
            "sender": "hack@fake.com",
            "subject": "Urgent security alert",
            "label": "mark_spam",
        },
        {
            "text": "Lunch with the team at 1 PM.",
            "sender": "hr@company.com",
            "subject": "Team lunch",
            "label": "escalate",
        },
    ],
    "medium": [
        {
            "text": "Claim your exclusive reward now using the link below.",
            "sender": "offers@deals.net",
            "subject": "Special offer for you",
            "label": "mark_spam",
        },
        {
            "text": "Project deadline moved to Friday. Please update your deliverables.",
            "sender": "pm@company.com",
            "subject": "Deadline update",
            "label": "escalate",
        },
        {
            "text": "New arrivals in electronics this week. Browse our latest picks.",
            "sender": "news@electronics.com",
            "subject": "This week's picks",
            "label": "promotions_tab",
        },
        {
            "text": "Verify your bank details immediately to avoid account suspension.",
            "sender": "support@bankfake.com",
            "subject": "Urgent: verify now",
            "label": "mark_spam",
        },
        {
            "text": "Your invoice #4521 is attached for review.",
            "sender": "billing@vendor.com",
            "subject": "Invoice #4521",
            "label": "escalate",
        },
        {
            "text": "Members-only discount inside. Limited-time fashion offer.",
            "sender": "vip@fashion.com",
            "subject": "VIP access unlocked",
            "label": "promotions_tab",
        },
        {
            "text": "Action required: contract renewal before month end.",
            "sender": "legal@company.com",
            "subject": "Contract renewal",
            "label": "escalate",
        },
        {
            "text": "You've been selected for a free trial. Start today.",
            "sender": "trial@software.io",
            "subject": "Start your free trial",
            "label": "promotions_tab",
        },
    ],
    "hard": [
        {
            "text": "Please confirm your payroll credentials immediately to prevent delayed salary processing.",
            "sender": "security@company-payroll.co",
            "subject": "Urgent payroll verification",
            "label": "mark_spam",
        },
        {
            "text": "Re: Board meeting moved to 6 PM. Updated agenda attached.",
            "sender": "ceo-office@company.com",
            "subject": "Board meeting update",
            "label": "escalate",
        },
        {
            "text": "Your shipment is delayed due to customs review. Track package here.",
            "sender": "shipping@dhl.com",
            "subject": "Shipment update",
            "label": "escalate",
        },
        {
            "text": "Flash sale: 70% off for the next 2 hours only. Buy now.",
            "sender": "flash@shopnow.com",
            "subject": "Flash Sale",
            "label": "promotions_tab",
        },
        {
            "text": "Verify your account details to avoid service suspension.",
            "sender": "noreply@paypa1.com",
            "subject": "Account suspension",
            "label": "mark_spam",
        },
        {
            "text": "Please review the attached client feedback before today's call.",
            "sender": "client@bigcorp.com",
            "subject": "Client feedback",
            "label": "escalate",
        },
        {
            "text": "Tax refund available. Submit your banking details to claim today.",
            "sender": "refund@gov-support.net",
            "subject": "Tax refund available",
            "label": "mark_spam",
        },
        {
            "text": "New discounts on items you viewed this week. Limited offer inside.",
            "sender": "recs@amazon-like.com",
            "subject": "Based on your browsing",
            "label": "promotions_tab",
        },
        {
            "text": "Team standup notes from today. Action items included below.",
            "sender": "scrum@company.com",
            "subject": "Standup notes",
            "label": "escalate",
        },
        {
            "text": "Last chance to grab our premium membership at 60% off.",
            "sender": "deals@trusted.com",
            "subject": "Limited time offer",
            "label": "promotions_tab",
        },
    ],
}

VALID_ACTIONS = ["mark_spam", "escalate", "promotions_tab"]
INTERNAL_DOMAINS = {"company.com", "bigcorp.com", "dhl.com", "vendor.com"}


class EmailEnv:
    def __init__(self) -> None:
        self.emails = []
        self.current_index = 0
        self.current_task = "easy"
        self.total_reward = 0.0
        self.predictions = []
        self.true_labels = []
        self.seed = None
        self.rng = random.Random()

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> Observation:
        if task not in DATASETS:
            raise ValueError(f"Invalid task '{task}'. Choose from: {', '.join(DATASETS.keys())}")

        self.current_task = task
        self.seed = seed
        self.rng = random.Random(seed)
        self.emails = DATASETS[task].copy()
        self.rng.shuffle(self.emails)
        self.current_index = 0
        self.total_reward = 0.0
        self.predictions = []
        self.true_labels = []
        return self._get_observation()

    def _penalty_for_misclassification(self, predicted: str, correct: str) -> tuple[float, str]:
        if correct == "escalate" and predicted == "mark_spam":
            return -2.5, "Critical miss: important email incorrectly sent to spam"
        if correct == "mark_spam" and predicted == "escalate":
            return -2.0, "Risky action: suspicious email incorrectly escalated"
        if correct == "escalate" and predicted == "promotions_tab":
            return -1.25, "Important email downgraded to low priority"
        if correct == "promotions_tab" and predicted == "mark_spam":
            return -0.75, "Promotion treated too aggressively as spam"
        if correct == "promotions_tab" and predicted == "escalate":
            return -0.5, "Promotion unnecessarily escalated"
        if correct == "mark_spam" and predicted == "promotions_tab":
            return -1.25, "Suspicious email softened into promotions tab"
        return -1.0, f"Wrong action. Chose '{predicted}', correct was '{correct}'"

    def step(self, action: str) -> StepResult:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Choose from: {', '.join(VALID_ACTIONS)}")

        if self.current_index >= len(self.emails):
            return StepResult(
                observation=None,
                reward=Reward(value=0.0, reason="Episode already done"),
                done=True,
                info={},
            )

        email = self.emails[self.current_index]
        correct = email["label"]

        if action == correct:
            reward_value = 2.0
            reason = f"Correct triage action: '{action}'"
        else:
            reward_value, reason = self._penalty_for_misclassification(action, correct)

        reward_value -= 0.1

        self.true_labels.append(correct)
        self.predictions.append(action)
        self.total_reward += reward_value
        self.current_index += 1

        done = self.current_index >= len(self.emails)

        if done:
            accuracy = sum(p == t for p, t in zip(self.predictions, self.true_labels)) / len(self.true_labels)
            if accuracy == 1.0:
                reward_value += 10.0
                self.total_reward += 10.0
                reason += " | PERFECT EPISODE BONUS +10"

        return StepResult(
            observation=self._get_observation() if not done else None,
            reward=Reward(value=round(reward_value, 2), reason=reason),
            done=done,
            info={
                "total_reward": round(self.total_reward, 2),
                "progress": f"{self.current_index}/{len(self.emails)}",
                "seed": self.seed,
            },
        )

    def state(self) -> dict:
        return {
            "task": self.current_task,
            "task_description": TASK_METADATA[self.current_task]["description"],
            "step": self.current_index,
            "total_emails": len(self.emails),
            "total_reward": round(self.total_reward, 2),
            "done": self.current_index >= len(self.emails),
            "seed": self.seed,
        }

    def get_grader_score(self) -> float:
        eps = 0.0001

        if not self.true_labels:
            return eps

        correct = sum(p == t for p, t in zip(self.predictions, self.true_labels))
        score = correct / len(self.true_labels)

        score = max(eps, min(score, 1.0 - eps))
        return round(score, 4)



    def _get_observation(self) -> Optional[Observation]:
        if self.current_index >= len(self.emails):
            return None

        email = self.emails[self.current_index]
        sender_domain = email["sender"].split("@")[-1].lower()
        text_blob = f"{email['subject']} {email['text']}".lower()
        urgency_words = ["urgent", "immediately", "today", "action required", "verify", "suspension", "deadline"]
        contains_links = any(word in email["text"].lower() for word in ["click", "link", "http", "verify here", "track package"])
        contains_urgency_words = any(word in text_blob for word in urgency_words)
        is_external_sender = sender_domain not in INTERNAL_DOMAINS

        return Observation(
            email_text=email["text"],
            sender=email["sender"],
            sender_domain=sender_domain,
            subject=email["subject"],
            contains_links=contains_links,
            contains_urgency_words=contains_urgency_words,
            is_external_sender=is_external_sender,
            step=self.current_index + 1,
            total_emails=len(self.emails),
            task=self.current_task,
            task_description=TASK_METADATA[self.current_task]["description"],
        )


app = FastAPI(title="Inbox Triage Action Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailEnv()


@app.get("/")
def home():
    return {
        "message": "Inbox Triage Action Environment is running",
        "endpoints": [
            "/reset?task=easy",
            "/step",
            "/state",
            "/tasks",
            "/grader",
            "/baseline",
            "/docs",
        ],
    }


@app.get("/reset")
def reset(task: str = "easy", seed: Optional[int] = None):
    try:
        return env.reset(task, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset_post(payload: Optional[ResetRequest] = None):
    try:
        task = payload.task if payload else "easy"
        seed = payload.seed if payload else None
        return env.reset(task, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: Action):
    try:
        return env.step(action.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy", "description": TASK_METADATA["easy"]["description"], "difficulty": "easy"},
            {"id": "medium", "description": TASK_METADATA["medium"]["description"], "difficulty": "medium"},
            {"id": "hard", "description": TASK_METADATA["hard"]["description"], "difficulty": "hard"},
        ],
        "action_schema": {
            "label": {
                "type": "string",
                "values": VALID_ACTIONS,
                "description": "Choose the inbox triage action: mark_spam, escalate, or promotions_tab",
            }
        },
    }


@app.get("/grader")
def grader():
    score = env.get_grader_score()
    correct = sum(p == t for p, t in zip(env.predictions, env.true_labels))
    total = len(env.true_labels)

    return {
        "score": score,
        "correct_predictions": correct,
        "total_predictions": total,
        "predictions": env.predictions,
        "true_labels": env.true_labels,
        "accuracy": f"{score * 100:.1f}%",
    }


@app.get("/baseline")
def baseline():
    results = {}
    for task in DATASETS:
        temp_env = EmailEnv()
        temp_env.reset(task, seed=42)
        done = False

        while not done:
            action = temp_env.rng.choice(VALID_ACTIONS)
            result = temp_env.step(action)
            done = result.done

        results[task] = {
            "score": temp_env.get_grader_score(),
            "total_reward": round(temp_env.total_reward, 2),
        }

    return {"baseline_scores": results}
