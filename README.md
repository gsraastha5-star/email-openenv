# Email Classifier RL Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent
learns to classify emails into **spam**, **important**, or **promotion**.

## Motivation
Email overload is a real-world problem. This environment trains agents to
triage emails automatically, with three difficulty levels.

## Action Space
| Action | Description |
|---|---|
| `spam` | Unsolicited/malicious email |
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
| easy | 5 | Obvious spam/ham |
| medium | 8 | Mixed, less obvious |
| hard | 10 | Tricky, ambiguous cases |

## Reward Logic
- `+2.0` correct classification
- `-1.0` wrong classification  
- `-0.1` step penalty (encourages efficiency)
- `+10.0` bonus for perfect episode

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Baseline
```bash
python inference.py
```

## Run API Server
```bash
uvicorn env:app --host 0.0.0.0 --port 7860
```

## API Endpoints
- `GET /reset?task=easy` — Start new episode
- `POST /step` — Take action `{"label": "spam"}`
- `GET /state` — Current environment state
- `GET /tasks` — List all tasks
- `GET /grader` — Get score for current episode
- `GET /baseline` — Run random baseline on all tasks
```

---

Your folder should now look like:
```
email-openenv/
├── env.py
├── inference.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md