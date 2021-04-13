"""
Text Analytics
"""

What are the different files
----------------------------
- submit_q0 -> Creating a .csv from the .json data and storing into the local project structure
- submit_q1 -> Reading the required data and performing certain preprocessing steps
- submit_q2 -> Reading required data and Performing sentence detection using Spacy
- submit_q3 -> Reading Required data and Performing word tokenization, lemmatization, part-of-speech tagging
- submit_q4 -> Building LDA model with 10 topics
- submit_q5 -> Building multinomial Naïve Bayes model
- submit_q6 -> Text summarization

Installing the required packages and libraries
----------------------------------------------
below are the required libraries that is used in this assignment

(Feel free to create a requirements.txt and install the
libraries and packages in a virtual env)
-----------------------------------------
blis==0.7.4
catalogue==2.0.1
certifi==2020.12.5
chardet==4.0.0
click==7.1.2
cycler==0.10.0
cymem==2.0.5
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl
gensim==3.8.1
idna==2.10
Jinja2==2.11.3
joblib==1.0.1
kiwisolver==1.3.1
MarkupSafe==1.1.1
matplotlib==3.4.1
murmurhash==1.0.5
nltk==3.5
numpy==1.20.2
packaging==20.9
pandas==1.2.3
pathy==0.4.0
Pillow==8.2.0
preshed==3.0.5
pydantic==1.7.3
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2021.1
regex==2021.3.17
requests==2.25.1
scikit-learn==0.24.1
scipy==1.6.2
six==1.15.0
sklearn==0.0
smart-open==3.0.0
spacy==3.0.5
spacy-legacy==3.0.1
srsly==2.4.0
summarization==0.0.1
thinc==8.0.2
threadpoolctl==2.1.0
tqdm==4.59.0
typer==0.3.2
urllib3==1.26.4
wasabi==0.8.2

How to run the files
--------------------

general command to run any given file
-------------------------------------
 python <filename.py>
--------------------

In order to generate the "output_data_cleaned.csv"
--------------------------------------------------
change the ".json" path to your local path in the file submit_q0.py
run the script and the .csv is generated

All the other files will be reading the data from the above .csv generated
Change the filename or add the path if the file is not in the project directory
