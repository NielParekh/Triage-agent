# Medical Triage Assistant

A two-agent AI-powered medical triage system that classifies patient symptoms and assigns urgency levels. Built with LiteLLM and powered by monitoring and testing via the litmus SDK.

## Overview

The system uses a two-stage pipeline:

1. **Symptom Extractor** — Parses free-text symptom descriptions into structured clinical context
2. **Triage Reasoner** — Reasons through the clinical picture and assigns urgency (Low/Medium/High)

### Urgency Levels

- **Low**: Mild symptoms, stable condition. Can be managed with self-care or routine GP appointment.
- **Medium**: Significant symptoms requiring prompt evaluation within hours (urgent care or same-day GP).
- **High**: Potentially life-threatening or rapidly worsening. Needs immediate emergency care (ER/911).

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key (or other LLM provider compatible with LiteLLM)

### Installation

1. Clone and navigate to the project:
```bash
cd /Users/devansh/Desktop/Triage-agent
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
export LITMUS_MODEL="gpt-4o-mini"  # or your preferred model
export LITMUS_TEMPERATURE="0.0"
```

You can also create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
LITMUS_MODEL=gpt-4o-mini
LITMUS_TEMPERATURE=0.0
```

## Usage

### Start the Server

To start the FastAPI server:

```bash
cd Triage-agent
source venv/bin/activate
python server.py
```

The server will be available at `http://localhost:8001`

### Command Line

Run the triage agent directly from the terminal:

```bash
python triage_agent.py "I have severe chest pain radiating to my left arm"
```

Output:
```
URGENCY: High

RECOMMENDATION:
Call 911 or go to the nearest emergency room immediately.

REASONING:
Chest pain radiating to the left arm is a classic presentation of acute coronary syndrome. This is a medical emergency requiring immediate intervention.

TOP CONCERNS:
  - Acute myocardial infarction (heart attack)
  - Angina pectoris
  - Aortic dissection

CAVEATS:
This assessment should not replace professional medical evaluation. Always seek immediate emergency care for chest pain.
```

### Python API

Import and use the triage functions in your own code:

```python
from triage_agent import triage, triage_to_str

# Get detailed result as dict
result = triage("I have a mild headache and slight fever")
print(f"Urgency: {result['urgency']}")
print(f"Recommendation: {result['recommendation']}")
print(f"Top Concerns: {result['top_concerns']}")

# Get formatted string output
output = triage_to_str("I have persistent cough for 2 weeks")
print(output)
```

#### Response Format

The `triage()` function returns a dict with:

```python
{
    "extracted": {
        "reported_symptoms": [...],
        "duration": "...",
        "severity_cues": [...],
        "red_flag_indicators": [...],
        "relevant_context": "...",
        "summary": "..."
    },
    "urgency": "Low|Medium|High",
    "recommendation": "...",
    "reasoning": "...",
    "top_concerns": [...],
    "caveats": "..."
}
```

### Golden Tests & Litmus CLI

The project includes a suite of golden tests defined in `litmus.toml`. These tests verify the model's behavior across different scenarios:

#### Initialize Litmus Project

Set up the litmus project locally:

```bash
litmus init
```

This initializes the litmus database and prepares your project for monitoring and golden test validation.

#### Watch Mode (Real-time Monitoring)

Monitor your triage agent in real-time with live updates on model performance and golden test results:

```bash
litmus watch
```

This command:
- Watches for calls to your triage agent
- Validates outputs against golden test patterns
- Displays real-time drift detection
- Shows model performance metrics
- Streams updates to your terminal as tests run

Run in one terminal while executing triage calls in another to see live results.

#### Manual Golden Test Runs

Run golden tests programmatically with the litmus SDK:

```python
from litmos_sdk import LitmosSDK

sdk = LitmosSDK()
# Golden tests will be automatically validated against your model responses
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `LITMUS_MODEL` | `gpt-4o-mini` | LLM model to use |
| `LITMUS_TEMPERATURE` | `0.0` | Model temperature (0 = deterministic) |

### litmus.toml

The `litmus.toml` file configures:

- **Project metadata** — ID, name, description
- **Agent entrypoint** — Points to `triage()` function
- **Settings** — Model and temperature
- **Golden tests** — Test cases for validation and drift detection

## Project Structure

```
triage_agent/
├── triage_agent.py       # Main agent code
├── litmus.toml           # Golden tests and configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in git)
├── README.md            # This file
└── litmus.db            # SQLite database for traces (generated)
```

## Dependencies

Key dependencies:

- **litellm** (1.82.0) — Unified LLM interface
- **litmus-sdk** (0.1.3) — Monitoring and drift detection with LLM as a judge
- **openai** (2.26.0) — OpenAI API client
- **python-dotenv** (1.2.2) — Environment variable management
- **click** (8.3.1) — CLI utilities
- **pydantic** (2.12.5) — Data validation

See `requirements.txt` for the full list.

## Monitoring & Drift Detection

The litmus SDK automatically tracks:

- All triage calls and their results
- Model outputs for drift analysis
- Comparison against golden test patterns

Database location: `./litmus.db` (SQLite)
Vector store location: `./chroma/` (ChromaDB)

## Disclaimer

**This tool is for educational and demonstration purposes only.** 

⚠️ **Do not rely on this system for actual medical decisions.** This AI-generated triage assessment:

- Cannot replace professional medical evaluation
- Is not a substitute for seeing a healthcare provider
- Should never delay seeking emergency care for serious symptoms
- Must always be validated by qualified medical professionals

For any serious symptoms, contact emergency services (911 in the US) or visit an emergency room immediately.

## API Documentation

### `triage(symptom_description, system_prompt=None) -> dict`

Runs the full 2-agent pipeline.

**Args:**
- `symptom_description` (str): Plain-English description of symptoms
- `system_prompt` (str, optional): Override default system prompt for drift testing

**Returns:**
- dict with keys: `extracted`, `urgency`, `recommendation`, `reasoning`, `top_concerns`, `caveats`

### `triage_to_str(symptom_description, system_prompt=None) -> str`

Convenience wrapper returning formatted string output. Used as litmus golden test entrypoint.
