# ElecTwit: A Framework for Studying Persuasion in Multi-Agent Social Systems

This repository contains the official code for the paper "ElecTwit: A Framework for Studying Persuasion in Multi-Agent Social Systems".

## Paper & Citation 

- Conference Paper: TBA 
- [Full paper](https://docs.google.com/document/d/1S8J58YVaPKM6BqbRy8oSFtTaXVd0M05E3ZVSKnsYb80/edit?usp=sharing)
## Core features 

- Multi-Agent Simulation: Simulates three types of agents:
  - Voters: Interact with the platform and vote for candidates.
  - Candidates: Compete for votes by posting persuasive content.
  - Eventor: Injects "news" events into the simulation to trigger agent responses.
- Realistic Social Environment: Reproduces key features of X (Twitter), including posts, replies (with a 280-character limit), and likes, all tracked with unique IDs.
- Complex Agent Backgrounds: Agents are modeled with "Big 5" personality traits and stances on six core political topics to simulate diverse perspectives.
- Persuasion Analysis: The framework is designed to capture all interactions (posts, replies, likes, votes) in structured JSON files for analyzing persuasive techniques.

## Getting Started 

1. Clone repository 

```
git clone https://github.com/tcmmichaelb139/ai-electwit.git
cd ai-electwit
```

2. Source and install 

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the simulation 

```
python main.py
```

The settings for the simulation can be edited in this file.

The analysis was done in Jupyter notebooks and can be viewed in `./analysis/`
