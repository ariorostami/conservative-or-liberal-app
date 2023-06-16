import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
import nltk

df = pd.read_csv("C:/Users/mario/Downloads/data folde/reddit_comments.csv", index_col=0)


from nltk import word_tokenize, sent_tokenize
df
data_frame = df.reset_index()

data_frame['Date Created'] = pd.to_datetime(data_frame['Date Created'],unit='s')
data_frame[['Text']] = data_frame[['Text']].fillna('NaN')

for index in data_frame.index:
    if data_frame.loc[index,'Text']!='NaN':
        data_frame.loc[index,'Text'] = data_frame.loc[index,'Title'] + "| " + data_frame.loc[index,'Text']
    else: data_frame.loc[index, 'Text'] =data_frame.loc[index,'Title']

X=data_frame["Text"]
y=data_frame["Political Lean"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

def prepareLabelData(data):
    # convert to 0s & 1s
    return pd.get_dummies(data)['Conservative'].to_numpy()
y_train = prepareLabelData(y_train)
y_test = prepareLabelData(y_test)

train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=123)
pipe_lr = make_pipeline(
    CountVectorizer(stop_words="english"),
    LogisticRegression(max_iter=2000),
)
pipe_lr.fit(X_train, y_train);
pipe_lr.score(X_train, y_train)
pipe_lr.score(X_test, y_test)

with open("web_api/moment_predictor.joblib", "wb") as f:
    joblib.dump(pipe_lr, f)
with open("web_application/moment_predictor.joblib", "wb") as f:
    joblib.dump(pipe_lr, f)

def return_prediction(model, text):
    prediction = model.predict([text])[0]
    return prediction

model = joblib.load("web_api/moment_predictor.joblib")

text = "I love Obama!"
return_prediction(model, text)

text = "Trump is the best president"
return_prediction(model, text)