# Support Ticket Search System

## Project Title
Support Ticket Search System — TF-IDF + Inverted Index

---

## Objective
A Flask-based web application that allows users to describe a support issue in plain English and instantly find the most relevant matching ticket from a knowledge base. The system uses TF-IDF vectorisation, an inverted index for candidate retrieval, and cosine similarity for ranking — providing transparent, explainable results at every phase of the search pipeline.

---

## Tools and Technologies
- **Python 3.x**
- **Flask** — web framework
- **scikit-learn** — TF-IDF vectorisation and cosine similarity
- **NLTK** — tokenisation, lemmatisation, stop word removal
- **NumPy** — vector operations
- **Gunicorn** — production WSGI server (for deployment)
- **HTML / CSS / JavaScript** — frontend interface

---

## Project Structure
```
project-folder/
├── app.py              ← Flask web server and route definitions
├── assign.py           ← Core NLP and IR engine (TF-IDF, inverted index)
├── tickets.txt         ← Ticket knowledge base (pipe-separated format)
├── requirements.txt    ← Python dependencies
├── README.md           ← Project documentation
├── .gitignore          ← Files excluded from version control
└── templates/
    └── index.html      ← Main HTML template
```

---

## Installation Steps

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. (Recommended) Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Ensure `tickets.txt` is present** in the same folder as `app.py`, formatted as:
```
TICKET_ID|Issue description text|Solution text
```

---

## How to Run the Application

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:8080
```

Type a support issue description into the search box and press **Search** (or `Ctrl+Enter`). The system will display ranked matching tickets with similarity scores and solution recommendations.

---

## How It Works

The search pipeline runs in four phases:

1. **Query Analysis** — tokenises the query, expands contractions, removes stop words, and applies lemmatisation + Porter stemming.
2. **Inverted Index Lookup** — uses root tokens to retrieve candidate tickets efficiently.
3. **Cosine Similarity Ranking** — computes TF-IDF weighted cosine similarity between the query and all candidates.
4. **Result Display** — returns top-K ranked results with descriptions, solutions, and per-token similarity breakdowns.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Renders the main search interface |
| POST | `/search` | Accepts `{ "query": "...", "top_k": 3 }`, returns ranked results as JSON |

---

## tickets.txt Format
Each line must follow this format (no line breaks within a ticket):
```
T001|Description of the issue|Solution steps
```

---

## Execution Screenshots
*(Add screenshots here after running the application)*

- Screenshot 1: Flask server running in terminal
- Screenshot 2: Application homepage in browser
- Screenshot 3: Search results displayed
- Screenshot 4: GitHub repository commit history

---

## Author
Ranjana Devi K  
Roll No: 2303121031
Department: Information Science and Engineering

## Project
Support Ticket Search — Flask Web Application

## Team Members
- Ranjana Devi K (Roll No: 2303121031)
- Ranjini R (Roll No: 2303121034)
- Subasri A(Roll No: 2303121041)
- Suvedha K(Roll No: 2303121043)
- Vaitheeshwari K(Roll No: 2303121045)