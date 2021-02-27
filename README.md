# Sentiment Classification APP  [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/phonhay103/SentimentClassificationSunAI)
Streamlit Web App to predict sentiment of a review

## Pre-requisites
The project was developed using python 3.8.8 with the following packages.
- Pandas
- Numpy
- Scikit-learn
- Pandas-profiling
- Joblib
- Streamlit
- Pandas_profiling
- Streamlit-pandas-profiling
- Tensorflow

Installation with pip:
```bash
pip install -r requirements.txt
```

## Getting Started
Open the terminal in you machine and run the following command to access the web application in your localhost.
```bash
streamlit run app.py
```

## Files
- Model.ipynb : Jupyter Notebook with all the workings including pre-processing, modelling and inference.
- app.py : Streamlit App script
- requirements.txt : pre-requiste libraries for the project
- models/ : trained model files and tokenizer objects
- images/ : background images

## Acknowledgements
[Streamlit](https://www.streamlit.io/), for the open-source library for rapid prototyping.
