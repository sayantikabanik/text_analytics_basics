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

### Example data format used for the scripts are
```{"reviewerID": "A30TL5EWN6DFXT", "asin": "120401325X", "reviewerName": "christina", "helpful": [0, 0], "reviewText": "They look good and stick good! I just don't like the rounded shape because I was always bumping it and Siri kept popping up and it was irritating. I just won't buy a product like this again", "overall": 4.0, "summary": "Looks Good", "unixReviewTime": 1400630400, "reviewTime": "05 21, 2014"}
{"reviewerID": "ASY55RVNIL0UD", "asin": "120401325X", "reviewerName": "emily l.", "helpful": [0, 0], "reviewText": "These stickers work like the review says they do. They stick on great and they stay on the phone. They are super stylish and I can share them with my sister. :)", "overall": 5.0, "summary": "Really great product.", "unixReviewTime": 1389657600, "reviewTime": "01 14, 2014"}```
