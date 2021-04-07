### Summary:

In this project we have the disaster_categories.csv and disaster_messages.csv. In this files the messages of disasters and the associated categories are
stored. This data is pre processed and stored in a sql database. With the data in the database a model is trained to get the categories of a disaster
message. As last step a html webpage is created. On that webpage you'll find some insights of the data and a textbox where you can input a message and
get the associated categories. This can help people or organizations during an event of a disaster. So they can for example filter out where and which 
help is required.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_data.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
   On Udacity VM you will find the webpage with the following steps: 
    - run 'env|grep WORK'
    - type in your browser: https://"SPACEID"-3001."SPACEDOMAIN"


### Files in repository
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md

### Installed packages:
Package                       Version    
----------------------------- -----------
json5                         0.8.5      
jsondiff                      1.1.1      
jsonpickle                    1.4.1      
Flask                         0.12.5       
hmsclient                     0.1.1      
html2text                     2018.1.9   
html5lib                      1.1        
nltk                          3.2.5
numpy                         1.12.1
pandas                        0.23.3 
pickleshare                   0.7.4
plotly                        2.0.15
scikit-image                  0.14.2     
scikit-learn                  0.19.1   
SQLAlchemy                    1.2.19
