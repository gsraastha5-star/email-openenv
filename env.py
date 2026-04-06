from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import random

# ─── Pydantic Models (OpenEnv typed spec) ────────────────────────────────────

class Observation(BaseModel):
    email_text: str
    sender: str
    subject: str
    step: int
    total_emails: int
    task: str

class Action(BaseModel):
    label: str  # "spam", "important", or "promotion"

class Reward(BaseModel):
    value: float
    reason: str

class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: dict

# ─── Email Datasets ───────────────────────────────────────────────────────────

DATASETS = {
    "easy": [
        {"text": "Win a free iPhone now! Click here!", "sender": "promo@scam.com",    "subject": "You won!!",                 "label": "spam"},
        {"text": "Team meeting at 5 PM today.",         "sender": "boss@company.com",  "subject": "Meeting reminder",          "label": "important"},
        {"text": "50% off on all shoes this weekend.",  "sender": "shop@store.com",    "subject": "Weekend Sale",              "label": "promotion"},
        {"text": "Your account has been compromised.",  "sender": "hack@fake.com",     "subject": "Urgent security alert",     "label": "spam"},
        {"text": "Lunch with the team at 1 PM.",        "sender": "hr@company.com",    "subject": "Team lunch",                "label": "important"},
    ],
    "medium": [
        {"text": "Claim your exclusive reward now.",           "sender": "offers@deals.net",   "subject": "Special offer for you",      "label": "spam"},
        {"text": "Project deadline moved to Friday.",          "sender": "pm@company.com",     "subject": "Deadline update",            "label": "important"},
        {"text": "New arrivals in electronics this week.",     "sender": "news@electronics.com","subject": "This week's picks",         "label": "promotion"},
        {"text": "Verify your bank details immediately.",      "sender": "support@bankfake.com","subject": "Urgent: verify now",        "label": "spam"},
        {"text": "Your invoice #4521 is attached.",            "sender": "billing@vendor.com", "subject": "Invoice #4521",              "label": "important"},
        {"text": "Members-only discount inside.",              "sender": "vip@fashion.com",    "subject": "VIP access unlocked",        "label": "promotion"},
        {"text": "Action required: contract renewal.",         "sender": "legal@company.com",  "subject": "Contract renewal",           "label": "important"},
        {"text": "You've been selected for a free trial.",     "sender": "trial@software.io",  "subject": "Start your free trial",      "label": "promotion"},
    ],
    "hard": [
        {"text": "Important update regarding your account security.", "sender": "security@bankreal.com", "subject": "Account security",       "label": "important"},
        {"text": "Last chance to grab our limited offer.",            "sender": "deals@trusted.com",     "subject": "Limited time offer",     "label": "promotion"},
        {"text": "Please review the attached client feedback.",       "sender": "client@bigcorp.com",     "subject": "Client feedback",        "label": "important"},
        {"text": "Verify your details to avoid suspension.",          "sender": "noreply@paypa1.com",     "subject": "Account suspension",     "label": "spam"},
        {"text": "New discounts on items you viewed.",                "sender": "recs@amazon-like.com",   "subject": "Based on your browsing", "label": "promotion"},
        {"text": "Urgent: your shipment is delayed.",                 "sender": "shipping@dhl.com",       "subject": "Shipment update",        "label": "important"},
        {"text": "You qualify for a government refund.",              "sender": "refund@gov-support.net", "subject": "Tax refund available",   "label": "spam"},
        {"text": "Team standup notes from today.",                    "sender": "scrum@company.com",      "subject": "Standup 2025-04-03",     "label": "important"},
        {"text": "Flash sale: 70% off for next 2 hours.",             "sender": "flash@shopnow.com",      "subject": "Flash Sale",             "label": "promotion"},
        {"text": "Congratulations, you won our survey prize.",        "sender": "prize@survey-win.com",   "subject": "Survey winner",          "label": "spam"},
    ],
}

VALID_ACTIONS = ["spam", "important", "promotion"]

# ─── Environment Class ────────────────────────────────────────────────────────

class EmailEnv:
    def __init__(self):
        self.emails = []
        self.current_index = 0
        self.current_task = "easy"
        self.total_reward = 0.0
        self.predictions = []
        self.true_labels = []

    def reset(self, task: str = "easy") -> Observation:
        self.current_task = task
        self.emails = DATASETS[task].copy()
        random.shuffle(self.emails)
        self.current_index = 0
        self.total_reward = 0.0
        self.predictions = []
        self.true_labels = []
        return self._get_observation()

    def step(self, action: str) -> StepResult:
        if self.current_index >= len(self.emails):
            return StepResult(
                observation=None,
                reward=Reward(value=0.0, reason="Episode already done"),
                done=True,
                info={}
            )

        email = self.emails[self.current_index]
        correct = email["label"]

        # Reward logic with partial signals
        if action == correct:
            reward_value = 2.0
            reason = f"Correct! '{action}' matched '{correct}'"
        else:
            reward_value = -1.0
            reason = f"Wrong. Chose '{action}', correct was '{correct}'"

        # Small step penalty to encourage efficiency
        reward_value -= 0.1

        self.true_labels.append(correct)
        self.predictions.append(action)
        self.total_reward += reward_value
        self.current_index += 1

        done = self.current_index >= len(self.emails)

        # Bonus for completing all correctly
        if done:
            accuracy = sum(p == t for p, t in zip(self.predictions, self.true_labels)) / len(self.true_labels)
            if accuracy == 1.0:
                reward_value += 10.0
                reason += " | PERFECT EPISODE BONUS +10"

        return StepResult(
            observation=self._get_observation() if not done else None,
            reward=Reward(value=round(reward_value, 2), reason=reason),
            done=done,
            info={
                "total_reward": round(self.total_reward, 2),
                "progress": f"{self.current_index}/{len(self.emails)}"
            }
        )

    def state(self) -> dict:
        return {
            "task": self.current_task,
            "step": self.current_index,
            "total_emails": len(self.emails),
            "total_reward": round(self.total_reward, 2),
            "done": self.current_index >= len(self.emails),
        }

    def get_grader_score(self) -> float:
        if not self.true_labels:
            return 0.0
        correct = sum(p == t for p, t in zip(self.predictions, self.true_labels))
        return round(correct / len(self.true_labels), 4)

    def _get_observation(self) -> Optional[Observation]:
        if self.current_index >= len(self.emails):
            return None
        e = self.emails[self.current_index]
        return Observation(
            email_text=e["text"],
            sender=e["sender"],
            subject=e["subject"],
            step=self.current_index + 1,
            total_emails=len(self.emails),
            task=self.current_task,
        )


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Email Classifier RL Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailEnv()

@app.get("/reset")
def reset(task: str = "easy"):
    obs = env.reset(task)
    return obs

@app.post("/step")
def step(action: Action):
    return env.step(action.label)

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy",   "description": "5 obvious emails",              "difficulty": "easy"},
            {"id": "medium", "description": "8 mixed emails",                "difficulty": "medium"},
            {"id": "hard",   "description": "10 tricky/ambiguous emails",    "difficulty": "hard"},
        ],
        "action_schema": {
            "label": {
                "type": "string",
                "values": VALID_ACTIONS,
                "description": "Classify the email as spam, important, or promotion"
            }
        }
    }

@app.get("/grader")
def grader():
    score = env.get_grader_score()
    return {
        "score": score,
        "predictions": env.predictions,
        "true_labels": env.true_labels,
        "accuracy": f"{score * 100:.1f}%"
    }

@app.get("/baseline")
def baseline():
    results = {}
    for task in ["easy", "medium", "hard"]:
        temp_env = EmailEnv()
        temp_env.reset(task)
        done = False
        while not done:
            action = random.choice(VALID_ACTIONS)
            result = temp_env.step(action)
            done = result.done
        results[task] = {
            "score": temp_env.get_grader_score(),
            "total_reward": temp_env.total_reward
        }
    return {"baseline_scores": results}