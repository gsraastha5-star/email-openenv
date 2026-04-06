# Email Classifier RL Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent learns to classify emails into `spam`, `important`, or `promotion`.

## Motivation

Email overload is a real-world problem. This environment trains agents to triage emails automatically across three difficulty levels.

## Action Space

| Action | Description |
|---|---|
| `spam` | Unsolicited or malicious email |
| `important` | Work-related or urgent email |
| `promotion` | Marketing or shopping email |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `email_text` | string | Body of the email |
| `sender` | string | Sender email address |
| `subject` | string | Email subject line |
| `step` | int | Current step number |
| `total_emails` | int | Total emails in episode |
| `task` | string | Current difficulty level |

## Tasks

| Task | Emails | Difficulty |
|---|---|---|
| `easy` | 5 | Obvious spam/ham |
| `medium` | 8 | Mixed, less obvious |
| `hard` | 10 | Tricky or ambiguous cases |

## Reward Logic

- `+2.0` for a correct classification
- `-1.0` for a wrong classification
- `-0.1` step penalty
- `+10.0` perfect episode bonus

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
