from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


nltk.download('stopwords')

app = Flask(__name__)

# Load the recall data and preprocess
data = pd.read_csv("recall.csv", dtype=str)
data = data[pd.notnull(data['Distribution Pattern'])]
df_1 = data[['Reason', 'Status']]
df_1 = df_1[df_1.Status != 'Completed'].reset_index(drop=True)

corpus = []
for i in range(0, 80613):
    review = re.sub('[^a-zA-Z]', ' ', df_1['Reason'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Create a CountVectorizer object to convert text to numerical features
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = df_1['Status'].apply(lambda x: 1 if x != 'Completed' else 0).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

# Train a Random Forest model
rfm = RandomForestClassifier(random_state=1)
rfm.fit(X_train, y_train)

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) # Define stop words globally

# Define the root route to handle user input
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['reason']
        print("User input:", user_input)
        review = re.sub('[^a-zA-Z]', ' ', user_input)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        X_user_input = cv.transform([review]).toarray()
        recall_reason_prediction = rfm.predict(X_user_input)[0]
        print("Prediction:", recall_reason_prediction)

        if recall_reason_prediction == 1:
            prediction_text = 'ongoing'
        else:
            prediction_text = 'terminated'
        return render_template('index.html', prediction_text=prediction_text)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
