# TMDB Hollywood ROI — AI Agent Project Instructions

## Purpose

You are working on an existing TMDB Hollywood ROI / Movie Revenue Prediction project.

Your job is to help turn the existing codebase into a clean, reproducible, portfolio-quality data science project **without unnecessarily rewriting working code**.

The project should follow a real data scientist workflow:

```text
Problem Definition
        ↓
Data Understanding / Validation
        ↓
EDA
        ↓
Feature Engineering
        ↓
Baseline
        ↓
Incremental Feature Experiments
        ↓
Model Comparison
        ↓
Error Analysis
        ↓
Final Pipeline
        ↓
Prediction / Streamlit App
        ↓
Documentation
```

The main objective is to predict movie revenue using information that would realistically be available at the defined prediction point.

---

# 1. CRITICAL RULE — PRESERVE THE EXISTING CODING STYLE

Before writing or changing code, inspect the existing repository.

**The existing code is the primary style reference.**

Do NOT automatically replace the current coding style with your preferred style.

Study existing files for:

- naming conventions
- function naming
- variable naming
- comments
- docstrings
- indentation
- imports
- configuration style
- database connection style
- logging style
- error handling
- API request patterns
- SQL style
- file organization
- how configuration/environment variables are loaded
- how dataframes are manipulated
- how results are saved
- how exceptions are handled

When adding new code, make it look like it was written by the same developer.

## Style-matching instruction

**Try to imitate the existing codebase rather than imposing a new coding philosophy.**

If existing code uses:

```python
def ingest_movie_details(movie_id):
    ...
```

continue using similarly named functions.

If existing code uses:

```python
log(...)
```

do not introduce a completely different logging framework unless there is a strong reason.

If existing code uses straightforward procedural Python, don't unnecessarily convert everything into classes.

If existing code uses pandas heavily, continue using pandas where appropriate.

If existing code has simple comments rather than extensive docstrings, don't add excessive documentation to every function.

If existing code has a particular database helper or utility, reuse it rather than creating another competing implementation.

---

# 2. DO NOT REWRITE WORKING CODE JUST FOR CLEANLINESS

The existing project may already contain substantial analytical/modeling work, including:

- TMDB API ingestion
- PostgreSQL database
- multiple database tables
- movie/person/credit data
- enrichment code
- existing utilities
- existing configuration
- existing scripts

Treat existing functionality as valuable.

Before modifying a file:

1. Read it.
2. Understand what it does.
3. Identify what depends on it.
4. Determine whether the change is actually necessary.
5. Make the smallest change that solves the problem.

Do NOT:

- rewrite the entire ingestion system
- rename everything
- move files unnecessarily
- replace working database code
- introduce frameworks without a reason
- create duplicate utilities
- create duplicate database connections
- change working API logic simply because you would implement it differently

---

# 3. ASK "WHY?" BEFORE ADDING CODE

Every significant new file, feature, model, dependency, or abstraction should have a reason.

Before adding something, ask:

> What problem does this solve?

Avoid:

```text
feature because it is possible
model because it is popular
library because it is commonly used
folder because the project looks cleaner
```

Prefer:

```text
problem
→ reason
→ implementation
→ measurable result
```

---

# 4. PROJECT STRUCTURE

The intended final structure is approximately:

```text
tmdb-hollywood-roi/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   └── config.py
│
├── data/
│   ├── README.md
│   └── sample/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── validate_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── historical_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   ├── figures/
│   └── experiments/
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    ├── test_preprocessing.py
    └── test_features.py
```

This is the **target structure**, not a requirement to create every file immediately.

---

# 5. BUILD THE REPOSITORY IN STAGES

Do not create the entire structure at once.

Only add a directory/file when the project reaches the stage where it is useful.

## Stage 1 — EDA

Use:

```text
notebooks/
└── 01_eda.ipynb
```

Purpose:

- understand the dataset
- inspect missing values
- inspect distributions
- inspect target variable
- identify suspicious values
- investigate relationships
- identify potential features

Do not put the entire project into this notebook.

---

## Stage 2 — Feature Engineering

Use:

```text
notebooks/
└── 02_feature_engineering.ipynb
```

Experiment with:

- numerical features
- categorical features
- release-date features
- genre features
- production features
- director features
- cast features
- crew features
- keyword features
- franchise features
- historical features

Once code becomes part of the actual pipeline, move reusable logic into:

```text
src/features/
```

The notebook is for exploration.

The `.py` files are for reusable implementation.

---

## Stage 3 — Data Layer

Add:

```text
src/data/
├── load_data.py
└── validate_data.py
```

Use these for the **ML/research repository's data access layer**, such as:

- loading the already-ingested TMDB data from PostgreSQL
- creating the ML dataset
- checking required columns
- checking duplicates
- validating target values
- checking unexpected nulls

The upstream TMDB ingestion pipeline lives in a **different repository**.

Do not copy ingestion code into this repository.

Do not create TMDB API ingestion scripts here unless explicitly requested.

Reuse only the data-access/database utilities that already exist in this repository where appropriate.

Do not create a second database connection system if one already exists.

---

## Stage 4 — Baseline

Create:

```text
notebooks/
└── 03_baseline.ipynb
```

Establish:

1. Naive baseline
2. Simple ML baseline
3. Initial XGBoost/gradient boosting model if appropriate

Record:

- MAE
- RMSE
- R²
- training configuration
- feature set
- observations

---

# 6. FIVE FEATURE-STAGE EXPERIMENTS

The project should initially evaluate feature groups incrementally.

Use the same underlying model and evaluation methodology.

### Model 1 — Basic

Possible features:

```text
budget
runtime
release_year
release_month
release_quarter
release_day_of_week
original_language
primary_genre
primary_production_country
```

### Model 2 — Production

Add:

```text
primary_production_company
production_company_count
production_country_count
```

### Model 3 — People

Add appropriate:

```text
director
cast_size
crew_size
```

and other justified people-related features.

### Model 4 — Historical

Add leakage-safe historical features such as:

```text
director_previous_movie_count
director_previous_avg_revenue
director_previous_avg_budget
director_previous_avg_roi
production_company_previous_avg_revenue
franchise_previous_avg_revenue
cast_previous_avg_revenue
```

### Model 5 — Advanced

Investigate:

```text
keywords
certification
franchise information
additional domain features
TMDB engagement variables
```

Only use engagement variables such as popularity/vote information if they are legitimately available at the defined prediction point.

---

# 7. KEEP THE FIVE MODELS COMPARABLE

Initially, keep the following constant:

- algorithm
- train/validation/test methodology
- random seed where applicable
- evaluation metrics
- core hyperparameters

The main thing changing should be the **feature set**.

This answers:

> "What does each additional group of features contribute?"

Do not simultaneously change:

```text
features
+
algorithm
+
hyperparameters
+
data split
```

because then it becomes difficult to understand why performance changed.

---

# 8. THEN COMPARE ALGORITHMS

After identifying a strong feature set, freeze that feature set.

Then compare algorithms such as:

```text
Linear Regression
Random Forest
XGBoost
LightGBM
```

Only include algorithms that make sense for the dataset.

The question at this stage is:

> "Which modeling approach works well with this feature set?"

Do not assume that the most complicated model is the best.

---

# 9. FEATURE ENCODING

Initial recommendations:

### `primary_genre`

Use:

```text
One-Hot Encoding
```

### `primary_production_country`

Use:

```text
One-Hot Encoding
```

Potentially use `min_frequency` to handle extremely rare categories.

### `primary_production_company`

Start with:

```text
Frequency Encoding
```

Investigate historical/target-style features only with careful leakage prevention.

### `director_name`

Do not simply one-hot thousands of directors.

Investigate:

```text
director_movie_count
director_previous_avg_revenue
director_previous_avg_budget
director_previous_avg_roi
```

These historical features MUST only use information available before the movie's prediction date.

---

# 10. DATA LEAKAGE IS A MAJOR CONCERN

Before using any feature, ask:

> Would this information actually be known when the prediction is made?

For example, if the project predicts revenue before release, do not use post-release information such as:

```text
opening_weekend_revenue
final_revenue
post-release vote count
post-release popularity
post-release audience information
```

Be especially careful with:

- target encoding
- director historical revenue
- actor historical revenue
- production-company historical revenue
- franchise historical revenue
- popularity
- vote count
- vote average

Historical features must be calculated using only prior records.

Example:

```text
Movie A — 2010
Movie B — 2015
Movie C — 2020

Prediction for Movie C:

Can use:
Movie A
Movie B

Cannot use:
Movie C revenue
```

---

# 11. TARGET VARIABLE

The primary target is expected to be:

```text
revenue
```

Before modeling:

- investigate missing revenue
- investigate zero revenue
- investigate extreme values
- inspect distribution
- consider log transformation
- document the decision

Do not blindly replace missing/zero revenue with zero without understanding what the value means.

---

# 12. ROI

ROI can be calculated as:

```python
roi = (revenue - budget) / budget
```

However:

**Do not use ROI as an input feature when predicting revenue.**

Revenue is part of the ROI calculation, so doing so creates target leakage.

ROI can instead be:

- an analysis metric
- a secondary target
- a historical feature based only on previous movies

---

# 13. EXPERIMENT TRACKING

Use:

```text
reports/
└── experiments/
    └── experiment_log.csv
```

Track experiments in a consistent format.

Suggested columns:

```text
experiment_id
feature_set
model
hyperparameters
train_period
validation_period
test_period
mae
rmse
r2
notes
```

Example:

```text
001 | basic       | XGBoost | ... | ... | ... | ... | ...
002 | production  | XGBoost | ... | ... | ... | ... | ...
003 | people      | XGBoost | ... | ... | ... | ... | ...
004 | historical  | XGBoost | ... | ... | ... | ... | ...
005 | advanced    | XGBoost | ... | ... | ... | ... | ...
```

Do not rely on notebook filenames to remember experiments.

---

# 14. TIME-AWARE VALIDATION

Because movie revenue is related to time, consider whether a random split accurately represents the real-world use case.

If the prediction is supposed to predict future movies, a time-based split may be more appropriate.

For example:

```text
Training:   older movies
Validation: later movies
Test:       newest movies
```

Do not blindly use a random train/test split if it creates an unrealistic prediction scenario.

---

# 15. ERROR ANALYSIS

Create:

```text
notebooks/
└── 05_error_analysis.ipynb
```

Investigate:

- largest overpredictions
- largest underpredictions
- blockbuster movies
- low-budget movies
- genres
- years
- languages
- franchises
- missing data
- unusual movies

The goal is to answer:

> "Why is the model wrong?"

not merely:

> "How do I increase R²?"

---

# 16. CODE ORGANIZATION

Once experimentation stabilizes:

```text
src/
├── data/
├── features/
├── models/
└── utils/
```

Keep functions focused.

Prefer:

```python
def create_release_features(df):
    ...
```

over putting the entire pipeline into one 1,000-line function.

But do not over-engineer.

A small portfolio project does not need dozens of classes and abstractions.

---

# 17. STREAMLIT

Only build the Streamlit app after the final prediction pipeline works.

Use:

```text
app/
└── streamlit_app.py
```

The app should call the same preprocessing and prediction functions used by the model.

Do NOT copy and paste preprocessing logic into Streamlit.

The intended flow is:

```text
User Input
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Trained Model
    ↓
Predicted Revenue
```

---

# 18. TESTING

Add:

```text
tests/
```

after reusable functions exist.

Start with simple tests for:

- preprocessing
- feature creation
- historical feature calculations
- handling missing values

Do not spend more time testing than building the actual project.

---

# 19. README

The final README should explain:

```text
Problem
Business Context
Dataset
Data Architecture
Feature Engineering
Prediction Point
Leakage Prevention
Modeling Methodology
Experiments
Results
Error Analysis
Streamlit Demo
Limitations
Future Improvements
How to Run
```

The README should make the project understandable to someone who has never seen the repository.

---

# 20. IMPORTANT AGENT BEHAVIOR

When asked to implement something:

### First

Inspect relevant existing code.

### Second

Explain briefly:

- what already exists
- what needs to change
- why the change is necessary

### Third

Make the smallest reasonable change.

### Fourth

Check that the new code is compatible with existing code.

### Fifth

If possible, run a relevant test or validation.

Do not make unrelated changes.

---

# 21. DO NOT DO THESE THINGS

Do not:

- rewrite working code for stylistic reasons
- create unnecessary abstractions
- create duplicate utility functions
- create duplicate database connections
- rename existing variables without a reason
- rename existing files without a reason
- add libraries unnecessarily
- create a new notebook for every tiny experiment
- blindly one-hot high-cardinality columns
- use target information in features
- use future information in historical features
- optimize for a metric without considering the real problem
- automatically choose a neural network because it sounds advanced
- delete existing work without checking its purpose
- silently change the project's prediction definition
- silently change the target variable
- silently change the data split
- silently change the meaning of existing columns
- recreate or move the TMDB ingestion pipeline into this repository
- modify the separate TMDB ingestion repository as part of normal work on this project

---

# 22. WHEN YOU ARE UNSURE

If there are multiple technically reasonable approaches:

1. Prefer the approach most consistent with the existing project.
2. Prefer the simpler approach.
3. Prefer the approach that is easier to reproduce.
4. Prefer the approach that makes the experiment easier to interpret.
5. If the choice materially changes the project's methodology, ask before implementing it.

Do not make large architectural decisions silently.

---

# 23. THE PROJECT'S CORE PRINCIPLE

Every experiment should answer a question.

Bad:

```text
experiment_17
try another model
```

Good:

```text
Question:
Does adding director historical performance improve
out-of-time revenue prediction?

Method:
Add leakage-safe director historical features.

Evaluation:
Compare against the previous feature set using
the same model and test period.

Result:
...

Conclusion:
...
```

The project should tell a coherent story.

---

# 24. DEFINITION OF DONE

The project is not finished simply because a model has a high R².

The final project should have:

- [ ] Clear problem definition
- [ ] Clearly defined prediction point
- [ ] Valid target
- [ ] Documented data sources
- [ ] Data validation
- [ ] EDA
- [ ] Leakage analysis
- [ ] Baseline
- [ ] Incremental feature experiments
- [ ] Algorithm comparison
- [ ] Error analysis
- [ ] Final model
- [ ] Reproducible preprocessing
- [ ] Saved model
- [ ] Prediction pipeline
- [ ] Streamlit application if appropriate
- [ ] Tests for important reusable functions
- [ ] Clean README
- [ ] Clean repository
- [ ] Documented limitations

---

# FINAL INSTRUCTION TO THE AI AGENT

Treat this as an existing project being gradually improved, **not a blank project that needs to be rebuilt from scratch**.

The developer's existing code, naming style, structure, data-access utilities, and implementation patterns in this repository should be treated as the source of truth whenever possible.

The separate TMDB ingestion repository is an upstream dependency, not part of this project.

**Read first. Understand second. Change third.**

When implementing new functionality, match the developer's existing coding style as closely as reasonably possible.

Do not optimize for making the code look like your own code.

Optimize for making the new code look like a natural continuation of the code that is already there.

The ultimate goal is a project that is:

```text
Simple
+
Reproducible
+
Explainable
+
Experiment-driven
+
Leakage-safe
+
Portfolio-quality
```

rather than unnecessarily complex.
