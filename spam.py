import click
import string


from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_SAVE = 'rf-clf.pkl'
TARGET_SAVE = 'target.pkl'
VECT_SAVE= 'vectorizer.pkl'

@click.group()
def cli():
    pass


@cli.command()
def format():
    pass


@cli.command()
@click.option('--data', default='dataset')
@click.option('--test_size', default=0.1)
@click.option('--save/--no-save', default=False)
def train(data, save, test_size):
    dataset = load_files(data)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=test_size)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=5,
                                  max_df = 0.8,
                                  sublinear_tf=True,
                                  use_idf=True)),
        ('clf', RandomForestClassifier())
    ])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    click.echo('Accuracy: {}'.format(accuracy))  
    if save:
        joblib.dump(dataset.target_names, TARGET_SAVE) 
        joblib.dump(clf, MODEL_SAVE) 
        joblib.dump(vectorizer, VECT_SAVE)
        click.echo(
            'Model saved in \'{}\'.\nVectorizer saved in \'{}\'.\nTarget names saved in \'{}\'.'.format(
                MODEL_SAVE, VECT_SAVE, TARGET_SAVE))


@cli.command()
@click.argument('document')
@click.option('--model', default=MODEL_SAVE)
@click.option('--vectorizer', default=VECT_SAVE)
@click.option('--target_names', default=TARGET_SAVE)
def query(document, model, vectorizer, target_names):
    clf = joblib.load(model)
    vect = joblib.load(vectorizer)
    targets = joblib.load(target_names)
    document_vector = vect.transform([document])
    click.echo(targets[clf.predict(document_vector)[0]])


@cli.command()
@click.option('--data', default='dataset')
@click.option('--selection', default='ExtraTrees')
@click.option('--test_size', default=0.1)
def feature_selection(data, selection, test_size):
    dataset = load_files(data)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=test_size)
    vectorizer = CustomVectorizer()
    train_vectors = vectorizer.fit(X_train)
    if selection in ('ExtraTrees', 'RandomForest'):
        if selection == 'ExtraTrees':
            model = ExtraTreesClassifier()
        elif selection == 'RandomForest':
            model = RandomForestClassifier()
        model.fit(train_vectors, y_train)
        results = sorted(zip(
            vectorizer.feature_names_, model.feature_importances_), 
            key=lambda tuple: tuple[1], reverse=True)
        for tuple in results:
            print('{}: {}'.format(tuple[0], tuple[1]))
    if selection in ('KBest'):
        model = SelectKBest(score_func=chi2)
        fit = model.fit(train_vectors, y_train)
        results = sorted(zip(vectorizer.feature_names_, fit.scores_), 
            key=lambda tuple: tuple[1], reverse=True)
        for tuple in results:
            print('{}: {:.4f}'.format(tuple[0], tuple[1]))
    else:
        raise ValueError('Unknown selection parameter.')


class CustomVectorizer(object):
    """docstring for CustomVectorizer"""
    def __init__(self):
        super(CustomVectorizer, self).__init__()
        self.feature_names_ = []

    def fit(self, raw_documents, y=None):
        X = []
        for document in raw_documents:
            features = {}
            document = self.preprocess(document)
            upper_chars = len(
                [letter for letter in document if letter.isupper()])
            lower_chars = len(
                [letter for letter in document if letter.islower()])
            features['words'] = len(document.split())
            features['upper_char_ratio'] = upper_chars / (
                upper_chars + lower_chars)
            features['lower_char_ratio'] = lower_chars / (
                upper_chars + lower_chars)
            vector = []
            feature_names = []
            for key in sorted(features.keys()):
                feature_names.append(key)
                vector.append(features[key])
            X.append(vector)
        self.feature_names_ = feature_names
        return X


    def preprocess(self, document):
        document = document.decode()
        document = document.replace('\n', ' ')
        translator = str.maketrans({key: None for key in string.punctuation})
        document = document.translate(translator)
        document = ''.join([i for i in document if not i.isdigit()])
        return document
        

if __name__ == '__main__':
    cli()
