---
title: Email-openEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Email Classifier RL Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent learns to classify emails into `spam`, `important`, or `promotion`.

## Motivation

Email overload is a real-world problem. This environment simulates email triage, a task that people actually perform in workplaces and personal inboxes every day. It helps evaluate whether an AI agent can prioritize useful messages, ignore spam, and recognize promotional content across different difficulty levels.

## Action Space

| Action | Description |
|---|---|
| `spam` | Unsolicited, suspicious, or malicious email |
| `important` | Work-related, urgent, or useful email |
| `promotion` | Marketing, discount, or shopping-related email |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `email_text` | string | Body text of the current email |
| `sender` | string | Sender email address |
| `subject` | string | Subject line of the email |
| `step` | int | Current step number in the episode |
| `total_emails` | int | Total emails in the current task |
| `task` | string | Current task difficulty (`easy`, `medium`, `hard`) |

## Tasks

| Task | Emails | Difficulty | Description |
|---|---|---|---|
| `easy` | 5 | Easy | Clearly identifiable spam, promotions, and important emails |
| `medium` | 8 | Medium | More mixed and less obvious email patterns |
| `hard` | 10 | Hard | Tricky, ambiguous, and more realistic cases such as phishing-like emails |

## Reward Logic

The environment gives reward at every step, not only at the end.

- `+2.0` for a correct classification
- `-1.0` for an incorrect classification
- `-0.1` step penalty to encourage efficiency
- `+10.0` bonus for a perfect episode

## Grading

Each task is graded programmatically using classification accuracy:


score = correct_predictions / total_predictions

## Baseline Results



Example output from `python inference.py`:


[START] task=easy env=email-openenv model=gpt-4o-mini
[STEP] step=1 action=important reward=1.90 done=false error=null
[STEP] step=2 action=important reward=-0.85 done=false error=null
[STEP] step=3 action=promotion reward=-1.10 done=false error=null
[STEP] step=4 action=spam reward=1.90 done=false error=null
[STEP] step=5 action=spam reward=-2.10 done=true error=null
[END] success=true steps=5 score=0.400 rewards=1.90,-0.85,-1.10,1.90,-2.10

[START] task=medium env=email-openenv model=gpt-4o-mini
[STEP] step=1 action=spam reward=-2.10 done=false error=null
[STEP] step=2 action=important reward=-0.85 done=false error=null
[STEP] step=3 action=important reward=-1.60 done=false error=null
[STEP] step=4 action=promotion reward=-1.10 done=false error=null
[STEP] step=5 action=spam reward=1.90 done=false error=null
[STEP] step=6 action=spam reward=-0.85 done=false error=null
[STEP] step=7 action=spam reward=-2.10 done=false error=null
[STEP] step=8 action=important reward=-0.85 done=true error=null
[END] success=true steps=8 score=0.125 rewards=-2.10,-0.85,-1.60,-1.10,1.90,-0.85,-2.10,-0.85

[START] task=hard env=email-openenv model=gpt-4o-mini
[STEP] step=1 action=promotion reward=1.90 done=false error=null
[STEP] step=2 action=promotion reward=-1.10 done=false error=null
[STEP] step=3 action=spam reward=-2.10 done=false error=null
[STEP] step=4 action=spam reward=-2.10 done=false error=null
[STEP] step=5 action=spam reward=1.90 done=false error=null
[STEP] step=6 action=important reward=-1.60 done=false error=null
[STEP] step=7 action=important reward=-0.85 done=false error=null
[STEP] step=8 action=promotion reward=1.90 done=false error=null
[STEP] step=9 action=spam reward=-2.10 done=false error=null
[STEP] step=10 action=important reward=-1.60 done=true error=null
[END] success=true steps=10 score=0.300 rewards=1.90,-1.10,-2.10,-2.10,1.90,-1.60,-0.85,1.90,-2.10,-1.60


# Setup Instructions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Usage Instructions
Run the API server:
uvicorn env:app --host 0.0.0.0 --port 7860

Open the local docs page:
http://127.0.0.1:7860/docs

Run the baseline inference script:
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"
python inference.py

Build and run with Docker:
docker build -t email-openenv .
docker run -p 7860:7860 email-openenv


