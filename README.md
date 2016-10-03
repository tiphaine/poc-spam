# poc-spam

Simple python script to train a RandomForestClassifier for spam/ham classification, model saving and querying.

I quickly wrote this to help out a friend who had spam issues on his website. This uses TF-IDF vectorizer only at the moment for features. Improved features will appear next if required.

## Install

`pip install -r requirements.txt`

## How to use

```
python spam.py train --data dataset --test_size 0.2 --save
```

To train a model using a sklearn formatted dataset in `dataset` using 0.20 of the data as hold out for tests and save the classifier, vectorizer, and target names.
  
```
python spam.py query 'is this spam or ham?'
```

To query the model for `is this spam or ham?` content.