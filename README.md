# Disaster Response Pipeline Project

<img width="946" alt="66" src="https://user-images.githubusercontent.com/55929646/144651657-b1eac0e7-8db9-47e4-830b-1e674d74be26.PNG">

### Description
In this project I tried to classified the disaster messages into categories. I build a model that classifies disaster messages and I used flask to show the results after the user enters a message.  

### Files in the repository

<ul>app
    <li>templates</li>
    This folder contains the html files go.html and master.html
    <li>run.py</li>
    This file will used to run teh flask which will used the model and database that we build before
</ul>

<ul>data
    <li>disaster_categories.csv</li>
    List of the categories that we will use
    <li>disaster_messages.csv</li>
    This file contains list of the messages
    <li>DisasterResponse.db</li>
    The database
    <li>CleanDatabase.db</li>
    The database after the cleaning process
    <li>process_data.py</li>
    This file contains the ETL Pipeline Preparation codes
    
</ul>

<ul>models
    <li>train_classifier.py</li>
    This file contains the ML Pipeline Preparation codes
    <li>cv_AdaBoost.pkl</li>
    Save the model as pickle file
</ul>


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
        <img width="789" alt="33" src="https://user-images.githubusercontent.com/55929646/144652505-9ae0f491-b8d7-47c4-874a-d0c0e673ffe9.PNG">

        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
        <img width="783" alt="44" src="https://user-images.githubusercontent.com/55929646/144652550-8e84c736-15d4-4319-ba97-68ab97514c83.PNG">


2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
    <img width="373" alt="55" src="https://user-images.githubusercontent.com/55929646/144652584-931db070-0100-48b4-9e21-1e4a1824c3cf.PNG">


3. Go to http://0.0.0.0:3001/


<img width="351" alt="22" src="https://user-images.githubusercontent.com/55929646/144652765-1e16f0ac-1130-4441-a52f-a55fa243e666.PNG">

<img width="714" alt="1" src="https://user-images.githubusercontent.com/55929646/144655315-0dc60527-4dd9-4cca-9a88-6da112e325eb.PNG">

<img width="665" alt="2" src="https://user-images.githubusercontent.com/55929646/144655321-1b6dd087-3121-485b-89c1-eecd6b9dbe5e.PNG">

<img width="666" alt="3" src="https://user-images.githubusercontent.com/55929646/144655326-72e61689-ed7e-4351-9745-d62c297e8004.PNG">

<img width="751" alt="4" src="https://user-images.githubusercontent.com/55929646/144655328-7749e445-0be6-4224-b477-03f74f324ea8.PNG">
