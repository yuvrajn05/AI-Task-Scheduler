# AI Scheduler Agent

This project implements an AI-powered task scheduling agent that assigns tasks to employees based on factors like skill matching, current workload, fatigue, and availability. The system includes an interactive user interface built with Streamlit and can intelligently match tasks using deep learning techniques and natural language processing.

---

## 🔧 Features

- 🧠 **AI-based Task Assignment** using DQN (Deep Q-Learning).
- 💬 **NLP Parsing of Task Descriptions** to extract requirements.
- 👷 **Employee Profiling** based on skillsets, availability, and fatigue.
- 📊 **Real-time Visualization** of assignments with Streamlit.
- ☁️ Can be deployed via Colab + ngrok for quick demos.

---

## 🖥️ Usage (via Google Colab)

1. Open the notebook `AI_Scheduler_Agent.ipynb` in Google Colab.
2. Run the cells to:
   - Install dependencies
   - Launch the Streamlit server
   - Tunnel the port using `pyngrok`
3. Access the public URL generated to use the app.

---

## 🚀 Running Locally

Use the app.py file to run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
streamlit run app.py
