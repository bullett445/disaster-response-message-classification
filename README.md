# Disaster Response Pipeline Project

### Project description

In this project a classifier is trained on the relevance of short messages which have been sent during a desaster.
The classes are given in multiple categories. The automated classification of messages shall help disaster response
teams to cope better with the high number of messages produced in situations like this. It can help avoid missing
messages which are important and require immediate attention.

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

### File descriptions

    README.md: this file
    etl\ exploratory code for preparing data and model
        ETL Pipeline Preparation.ipynb: jupyter notebook exploring the pipeline
        ML Pipeline Preparation.ipynb: jupyter notebook exploring the model
    webapp\ plain python implementation
        app\
            run.py: main routine running the flask app
            templates\
                master.html: home page template 
                go.html: query result template
        data\
            disaster_categories.csv: categories dataset
            disaster_messages.csv: messages dataset
            DisasterResponse.db: disaster response database
            process_data.py: ETL process
        models\
            train_classifier.py: build and optimize the model

### Acknowledgements
This project was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/). Based
on a dataset provided by [Figure Eight](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) 
