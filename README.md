# Interview Chatbot

A web-based interview chatbot for research with dysarthria populations, built on top of [SparkMe](https://github.com/SALT-NLP/SparkMe).

## Attribution

This project is derived from **SparkMe** by David Anugraha, Vishakh Padmakumar, and Diyi Yang (SALT Lab, Stanford NLP).

- Original repository: https://github.com/SALT-NLP/SparkMe
- Original paper: [SparkMe: Adaptive Semi-Structured Interviewing for Qualitative Insight Discovery](https://arxiv.org/abs/2602.21136)
- Original license: Apache 2.0

### Changes made in this fork

- Added `streamlit_app.py`: a Streamlit web interface that wraps the SparkMe interview engine, replacing the original terminal and GCP-based web interfaces
- Added Google Drive integration: session logs and AI-generated notes are automatically uploaded to a private Google Drive folder at the end of each interview
- Added `web_app.py`: an alternative Flask-based local web interface (not used in deployment)
- Updated `requirements.txt` to include Streamlit and Google Drive API dependencies
- Updated `.gitignore` to exclude runtime participant data (`logs/`, `data/*/`, `.streamlit/`)

## Setup

### Requirements

- Python 3.10+
- An OpenAI API key
- A Google Cloud service account with Google Drive API access (for saving session data)

### Local development

```bash
git clone https://github.com/ZhuoanZhou/interview-chatbot.git
cd interview-chatbot
pip install -r requirements.txt
cp .env_sample .env          # fill in OPENAI_API_KEY
streamlit run streamlit_app.py
```

### Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io). Secrets (OpenAI key, Google Drive credentials) are stored in Streamlit's encrypted Secrets manager and never committed to this repository.

Session data is saved privately to Google Drive and is not accessible to the public.

## License

This project is licensed under the [Apache License 2.0](LICENSE), consistent with the original SparkMe project.

## Citation

If you use SparkMe's underlying system in your research, please cite the original work:

```bibtex
@article{anugraha2026sparkme,
  title={SparkMe: Adaptive Semi-Structured Interviewing for Qualitative Insight Discovery},
  author={Anugraha, David and Padmakumar, Vishakh and Yang, Diyi},
  journal={arXiv preprint arXiv:2602.21136},
  year={2026}
}
```
