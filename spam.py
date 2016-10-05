import click


from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
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
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size)
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
@click.argument('msg')
@click.option('--model', default=MODEL_SAVE)
@click.option('--vectorizer', default=VECT_SAVE)
@click.option('--target_names', default=TARGET_SAVE)
def query(msg, model, vectorizer, target_names):
    clf = joblib.load(model)
    vect = joblib.load(vectorizer)
    targets = joblib.load(target_names)
    msg_vector = vect.transform([msg])
    click.echo(targets[clf.predict(msg_vector)[0]])

if __name__ == '__main__':
    cli()
