# 🎧 Jan 2026 DLGenAI Project — Messy Mashup

**Predicting Music Genre from Noisy Mashups**

---

## 📌 Project Overview

This project is part of the **Jan 2026 Deep Learning & Generative AI (DLGenAI) Project**.

The goal is to **predict the correct music genre** of a noisy audio mashup.
Each mashup is created by mixing instrument stems (drums, vocals, bass, others) from different songs of the same genre, with additional tempo changes and random noise.

The task focuses on building models that are **robust to noise, tempo variation, and stem recombination**, similar to real-world music classification problems.

---

## 🧠 Problem Statement

Given a noisy audio mashup, predict one of the following genres:

```
["blues", "classical", "country", "disco", "hiphop",
 "jazz", "metal", "pop", "reggae", "rock"]
```

The main challenge is that:

* Training and test data come from **different distributions**
* Test samples include **noise and tempo variations**
* Models must learn **genre-level musical patterns**, not just clean audio features

---

## 📊 Evaluation Metric

Submissions are evaluated using **Macro F1 Score** across all 10 genres.

* F1 score is computed per genre
* Final score is the average across genres
* All genres are weighted equally

---

## 🗂 Repository Status (Initial)

This repository is currently in the **initial setup stage**.

Planned contents:

* Exploratory analysis notebooks
* Training code for multiple models
* Experiment tracking with Weights & Biases
* Final inference notebook for Kaggle
* Technical report

Code, models, and results will be added progressively as the project advances.

---

## 🚧 Current Status

* [x] Repository initialized
* [ ] Dataset exploration
* [ ] Baseline model
* [ ] Deep learning models
* [ ] Kaggle submissions

---

*This README will be updated as the project progresses.*
