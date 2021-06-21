#### Text Analytics (preprocessing and simple models)
Scripts related to basics data preprocessing and models like LDA, MNB 

#### What are the different files
```
- submit_q0 -> Creating a .csv from the .json data and storing into the local project structure
- submit_q1 -> Reading the required data and performing certain preprocessing steps
- submit_q2 -> Reading required data and Performing sentence detection using Spacy
- submit_q3 -> Reading Required data and Performing word tokenization, lemmatization, part-of-speech tagging
- submit_q4 -> Building LDA model with 10 topics
- submit_q5 -> Building multinomial Naïve Bayes model
- submit_q6 -> Text summarization
```

#### Installing the required packages and libraries
(Use the requirements.txt and install the
libraries and packages in a virtual env)

#### General command to run any given file
 ```python <filename.py>```

#### In order to generate the "output_data_cleaned.csv"
change the ```"data_cell.json"``` path to your local path in the file ```submit_q0.py```
run the script and the ```.csv``` is generated

All the other files will be reading the data from the above ```.csv``` generated
Change the filename or add the path if the file is not in the project directory
