# 🏓 Ping Pong Analytics Dashboard

A machine learning pipeline for analyzing doubles ping pong games using
Elo ratings, TrueSkill, and Gradient Boosted Decision Trees.

The project predicts match outcomes, quantifies player chemistry,
explores corner effects, explains model predictions using SHAP,
and generates an interactive HTML dashboard.

---

## Features

### Rating Systems

- Elo rating system
- TrueSkill Bayesian rating system
- Margin-of-victory Elo updates
- Elo and TrueSkill leadership tracking
- Rating history over time

### Machine Learning

- Gradient Boosted Decision Tree classifier
- Time-series cross validation
- SHAP feature explanations
- Feature importance ranking
- Ablation studies
- Prediction calibration metrics
- Brier score analysis

### Feature Engineering

More than 50 engineered features including:

- Elo
- TrueSkill
- Rating momentum
- Rating confidence
- Rating disagreement
- Player experience
- Recent form
- Team chemistry
- Partnership history
- Opponent familiarity
- Corner specialization
- Corner strength
- Team balance
- Team specialization
- Matchup history

### Interactive Dashboard

The generated dashboard includes:

- Player rankings
- Elo analysis
- TrueSkill analysis
- Machine learning analysis
- Partnership network graph
- Pair synergy plots
- Corner heatmaps
- Prediction heatmaps
- SHAP visualizations
- Feature importance
- Cross validation
- Game explorer
- Interactive matchup predictor

---

# Example Dashboard

![Dashboard](images/dashboard.png)

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pingpong-dashboard.git
cd pingpong-dashboard
```

Install required packages

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scipy
pip install scikit-learn
pip install plotly
pip install networkx
pip install trueskill
pip install shap
```

---

# Running

Edit

```python
INPUT_FILE
OUTPUT_DIR
```

inside

```
PingPongPlayerAnalysis3_Current.py
```

Then run

```bash
python PingPongPlayerAnalysis3_Current.py
```

or simply press **F5** inside IDLE.

---

# Input Format

The parser accepts a simple text log.

Example

```
TT, NS Vs. SY, CH: 21-18
FZ, AC Vs. TD, RS: 21-12
```

Blank lines separate days.

---

# Outputs

The pipeline automatically generates

```
dashboard.html

Player summaries

Elo rankings

TrueSkill rankings

Machine Learning metrics

Feature importance

SHAP analysis

Prediction heatmaps

Pair synergy

Leadership timelines

Interactive matchup explorer

CSV tables

PNG figures
```

---

# Models

## Elo

Traditional rating system

Uses

- expected win probability
- margin of victory
- K-factor updates

---

## TrueSkill

Bayesian skill estimation

Tracks

- mean skill (μ)
- uncertainty (σ)

Updates player skill distributions after every match.

---

## Gradient Boosted Trees

Predicts doubles outcomes using engineered features.

Advantages

- nonlinear relationships
- feature interactions
- interpretable using SHAP
- probability predictions

---

# Dashboard

The interactive dashboard contains six major sections.

- Tools
- General Statistics
- Elo Model
- TrueSkill Model
- Gradient Boosted Trees
- Model Comparisons

---

# Current Dataset

Current league

- 8 players
- 210 unique games
- 1680 ordered corner configurations
- 28 possible teammate pairs

---

# Repository Structure

```
PingPongPlayerAnalysis3_Current.py

PingPongData.txt

ping_pong_output/

images/

README.md
```

---

# Future Work

- Automatic GitHub Pages deployment
- Live dashboard updates
- Tournament simulation
- Match recommendation engine
- Hyperparameter optimization
- Additional rating systems
- Active learning for matchup selection

---

# Author

Cole Harris

PhD Student

University of Southern California

---

# License

MIT License
