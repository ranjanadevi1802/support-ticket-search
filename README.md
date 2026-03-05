
# Support Ticket Search System

Support Ticket Search System — TF-IDF + Inverted Index
A Flask-based web application that allows users to describe a support issue in plain English and instantly find the most relevant matching ticket from a knowledge base. The system uses TF-IDF vectorisation, an inverted index for candidate retrieval, and cosine similarity for ranking — providing transparent, explainable results at every phase of the search pipeline.



## About Me
Author:
Ranjana Devi K  
Roll No: 2303121031   
Department: Information Science and Engineering   

                                             
## Live Demo
https://support-ticket-search.onrender.com


## Features

-Natural language search for support tickets  
-TF-IDF vectorization for document representation  
-Inverted index for efficient search   
-Cosine similarity ranking  
-Web interface using Flask  
-Displays ranked results with similarity scores 

                
## Technologies used in the project:

Python 3.x   
Flask — web framework  
scikit-learn — TF-IDF vectorisation and cosine similarity  
NLTK — tokenisation, lemmatisation, stop word removal  
NumPy — vector operations  
Gunicorn — production WSGI server (for deployment)  
HTML / CSS / JavaScript — frontend interface


## Installation

1.Ensure Python 3.x is installed on your system.   
2.Create a virtual environment (recommended)

```bash
python -m venv venv
```
3.Activate the virtual environment  
Windows
```bash
venv\Scripts\activate
```
Mac / Linux
```bash
source venv/bin/activate
```
4.Install the required dependencies
```bash
pip install -r requirements.txt
```

    
## Run Locally

Clone the project

```bash
  git clone https://github.com/ranjanadevi1802/support-ticket-search.git
```

Go to the project directory

```bash
 cd support-ticket-search
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```

Open your browser and visit

```bash
  http://localhost:8080
```

The Support Ticket Search System interface will appear, where you can enter a query and retrieve relevant support tickets.
## Deployment

The application is deployed using Render with Gunicorn as the production server.

Live deployment:

https://support-ticket-search.onrender.com


## API Reference

| Method | Route     | Description                                     |
| ------ | --------- | ----------------------------------------------- |
| GET    | `/`       | Displays the search interface                   |
| POST   | `/search` | Accepts search query and returns ranked tickets |

Example request

{
 "query": "internet not working",
 "top_k": 3
}

## Usage/Examples

Example ticket format in tickets.txt

T001|Internet connection not working|Restart router and check cable connections

### Example query
wifi not connecting

The system returns the most relevant support tickets.
## Documentation

Project Structure

project-folder/   
├── app.py              # Flask web server and route definitions  
├── assign.py           # Core NLP and IR engine (TF-IDF, inverted index)  
├── tickets.txt         # Ticket knowledge base (pipe-separated format)  
├── requirements.txt    # Python dependencies  
├── README.md           # Project documentation  
├── .gitignore          # Files excluded from version control  
└── templates/
    └── index.html      # Main HTML template  


## Screenshots

### 1. Application Homepage (Live)
![Homepage](https://raw.githubusercontent.com/ranjanadevi1802/support-ticket-search/main/screenshots/homepage.png)

### 2. Search Results
![Results](https://raw.githubusercontent.com/ranjanadevi1802/support-ticket-search/main/screenshots/results.png)

## Authors

Ranjana Devi K  
Department: Information Science and Engineering

## Support

If you find this project useful or want to improve it, feel free to open an issue or contribute.

## License

This project is for educational and academic purposes.

