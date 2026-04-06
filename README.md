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


Final Score (easy): 0.20 (20.0% accuracy)
Final Score (medium): 0.25 (25.0% accuracy)
Final Score (hard): 0.20 (20.0% accuracy)

BASELINE SUMMARY
easy   : 0.20
medium : 0.25
hard   : 0.20
