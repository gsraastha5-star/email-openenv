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

<<<<<<< HEAD
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


## Baseline Results

The baseline agent was run using `inference.py` with the configured model and API settings.

==================================================
TASK: EASY
==================================================
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 1:
  Subject: Team lunch
  Sender : hr@company.com
  Email  : Lunch with the team at 1 PM.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 2:
  Subject: Meeting reminder
  Sender : boss@company.com
  Email  : Team meeting at 5 PM today.
  Chose  : important
  Reward : +1.9 (Correct! 'important' matched 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 3:
  Subject: You won!!
  Sender : promo@scam.com
  Email  : Win a free iPhone now! Click here!
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 4:
  Subject: Urgent security alert
  Sender : hack@fake.com
  Email  : Your account has been compromised.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 5:
  Subject: Weekend Sale
  Sender : shop@store.com
  Email  : 50% off on all shoes this weekend.
  Chose  : important
  Reward : -1.1 (Wrong. Chose 'important', correct was 'promotion')

Final Score (easy): 0.20 (20.0% accuracy)
Total Reward: -2.50

==================================================
TASK: MEDIUM
==================================================
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 1:
  Subject: Deadline update
  Sender : pm@company.com
  Email  : Project deadline moved to Friday.
  Chose  : important
  Reward : +1.9 (Correct! 'important' matched 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 2:
  Subject: Special offer for you
  Sender : offers@deals.net
  Email  : Claim your exclusive reward now.
  Chose  : important
  Reward : -1.1 (Wrong. Chose 'important', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 3:
  Subject: VIP access unlocked
  Sender : vip@fashion.com
  Email  : Members-only discount inside.
  Chose  : spam
  Reward : -1.1 (Wrong. Chose 'spam', correct was 'promotion')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 4:
  Subject: Invoice #4521
  Sender : billing@vendor.com
  Email  : Your invoice #4521 is attached.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 5:
  Subject: Contract renewal
  Sender : legal@company.com
  Email  : Action required: contract renewal.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 6:
  Subject: Urgent: verify now
  Sender : support@bankfake.com
  Email  : Verify your bank details immediately.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 7:
  Subject: Start your free trial
  Sender : trial@software.io
  Email  : You've been selected for a free trial.
  Chose  : spam
  Reward : -1.1 (Wrong. Chose 'spam', correct was 'promotion')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 8:
  Subject: This week's picks
  Sender : news@electronics.com
  Email  : New arrivals in electronics this week.
  Chose  : promotion
  Reward : +1.9 (Correct! 'promotion' matched 'promotion')

Final Score (medium): 0.25 (25.0% accuracy)
Total Reward: -2.80

==================================================
TASK: HARD
==================================================
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 1:
  Subject: Limited time offer
  Sender : deals@trusted.com
  Email  : Last chance to grab our limited offer.
  Chose  : spam
  Reward : -1.1 (Wrong. Chose 'spam', correct was 'promotion')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 2:
  Subject: Based on your browsing
  Sender : recs@amazon-like.com
  Email  : New discounts on items you viewed.
  Chose  : promotion
  Reward : +1.9 (Correct! 'promotion' matched 'promotion')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 3:
  Subject: Survey winner
  Sender : prize@survey-win.com
  Email  : Congratulations, you won our survey prize.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 4:
  Subject: Account security
  Sender : security@bankreal.com
  Email  : Important update regarding your account security.
  Chose  : important
  Reward : +1.9 (Correct! 'important' matched 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 5:
  Subject: Standup 2025-04-03
  Sender : scrum@company.com
  Email  : Team standup notes from today.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 6:
  Subject: Tax refund available
  Sender : refund@gov-support.net
  Email  : You qualify for a government refund.
  Chose  : important
  Reward : -1.1 (Wrong. Chose 'important', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 7:
  Subject: Client feedback
  Sender : client@bigcorp.com
  Email  : Please review the attached client feedback.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 8:
  Subject: Account suspension
  Sender : noreply@paypa1.com
  Email  : Verify your details to avoid suspension.
  Chose  : important
  Reward : -1.1 (Wrong. Chose 'important', correct was 'spam')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 9:
  Subject: Shipment update
  Sender : shipping@dhl.com
  Email  : Urgent: your shipment is delayed.
  Chose  : promotion
  Reward : -1.1 (Wrong. Chose 'promotion', correct was 'important')
[LLM error: Error code: 401 - {'error': {'message': 'Incorrect API key provided: dummy. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}] using random fallback

Step 10:
  Subject: Flash Sale
  Sender : flash@shopnow.com
  Email  : Flash sale: 70% off for next 2 hours.
  Chose  : spam
  Reward : -1.1 (Wrong. Chose 'spam', correct was 'promotion')

Final Score (hard): 0.20 (20.0% accuracy)
Total Reward: -5.00

==================================================
BASELINE SUMMARY
==================================================
  easy    : 0.20  ####
  medium  : 0.25  #####
  hard    : 0.20  ####

