# Disaster Response Pipeline Project

### Required libraries
* nltk
* flask
* plotly
* sqlalchemy
* sklearn
* pandas

If you also want to run the notebooks, you also need
* seaborn
* matplotlib

### Instructions:
1. Run the following commands in the webapp/ directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl` Depending
   on your computing power that may take a moment or two.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
