# Inbox Triage Action Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent learns to take realistic inbox triage actions such as `mark_spam`, `escalate`, or `promotions_tab`.

## Motivation

Email overload is a real-world problem in both personal and workplace inboxes. This environment simulates a more realistic inbox assistant workflow: instead of only classifying messages, the agent must decide what operational action should be taken on each email, such as escalating important messages, routing promotions away from the main inbox, or marking suspicious emails as spam.

## Action Space

| Action | Description |
|---|---|
| `mark_spam` | Route suspicious, scam, phishing, or malicious emails to spam |
| `escalate` | Prioritize and surface urgent, important, or operational emails |
| `promotions_tab` | Route low-priority commercial and marketing emails to promotions |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `email_text` | string | Body text of the current email |
| `sender` | string | Sender email address |
| `sender_domain` | string | Sender email domain |
| `subject` | string | Subject line of the email |
| `contains_links` | boolean | Whether the email appears to contain a link or call-to-click |
| `contains_urgency_words` | boolean | Whether the email contains urgent or high-pressure wording |
| `is_external_sender` | boolean | Whether the sender is outside trusted internal domains |
| `step` | int | Current step number in the episode |
| `total_emails` | int | Total emails in the current task |
| `task` | string | Current task difficulty (`easy`, `medium`, `hard`) |
| `task_description` | string | Human-readable description of the task |

## Tasks

| Task | Emails | Difficulty | Description |
|---|---|---|---|
| `easy` | 5 | Easy | Personal inbox triage with obvious spam, promotions, and important messages |
| `medium` | 8 | Medium | Workplace inbox triage with invoices, deadlines, renewals, and mixed-intent emails |
| `hard` | 10 | Hard | Security-sensitive inbox triage with phishing-like, ambiguous, and high-stakes business emails |

## Reward Logic

The environment gives reward at every step, not only at the end.

- `+2.0` for a correct triage action
- stronger negative rewards for unsafe or costly mistakes
- `-0.1` step penalty to encourage efficient behavior
- `+10.0` bonus for a perfect episode

Examples of asymmetric penalties:
- marking an important email as spam receives a stronger penalty
- escalating a suspicious phishing email also receives a strong penalty
- misrouting a promotion is penalized less severely than missing an important message

## Grading

Each task is graded programmatically using triage accuracy:

text
score = correct_predictions / total_predictions

## Setup Instructions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Usage Instructions
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

## Baseline Results

Example output from `python inference.py`:


[START] task=easy env=email-openenv model=gpt-4o-mini
[STEP] step=1 action=mark_spam reward=1.90 done=false error=null
[STEP] step=2 action=escalate reward=1.90 done=false error=null
[STEP] step=3 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=4 action=escalate reward=1.90 done=false error=null
[STEP] step=5 action=mark_spam reward=11.90 done=true error=null
[END] success=true steps=5 score=0.9999 rewards=1.90,1.90,1.90,1.90,11.90

[START] task=medium env=email-openenv model=gpt-4o-mini
[STEP] step=1 action=mark_spam reward=1.90 done=false error=null
[STEP] step=2 action=escalate reward=1.90 done=false error=null
[STEP] step=3 action=escalate reward=1.90 done=false error=null
[STEP] step=4 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=5 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=6 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=7 action=promotions_tab reward=-1.35 done=false error=null
[STEP] step=8 action=escalate reward=1.90 done=true error=null
[END] success=true steps=8 score=0.8750 rewards=1.90,1.90,1.90,1.90,1.90,1.90,-1.35,1.90

[START] task=hard env=email-openenv model=gpt-4o-mini
[STEP] step=1 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=2 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=3 action=escalate reward=1.90 done=false error=null
[STEP] step=4 action=escalate reward=1.90 done=false error=null
[STEP] step=5 action=escalate reward=1.90 done=false error=null
[STEP] step=6 action=mark_spam reward=1.90 done=false error=null
[STEP] step=7 action=promotions_tab reward=1.90 done=false error=null
[STEP] step=8 action=mark_spam reward=1.90 done=false error=null
[STEP] step=9 action=mark_spam reward=1.90 done=false error=null
[STEP] step=10 action=escalate reward=11.90 done=true error=null
[END] success=true steps=10 score=0.9999 rewards=1.90,1.90,1.90,1.90,1.90,1.90,1.90,1.90,1.90,11.90
