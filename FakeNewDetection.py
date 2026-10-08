from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data
news = [
    "Government announces new education scheme",
    "Scientists discover water on Mars",
    "President meets foreign leaders",
    "New technology improves mobile battery life",
    "Aliens landed in a village yesterday",
    "Drinking petrol can cure all diseases",
    "Moon is made of cheese",
    "A person can live without food for one year"
]

# 1 = Real, 0 = Fake
result = [1, 1, 1, 1, 0, 0, 0, 0]

# Convert text into numbers
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(news)

# Train model
model = LogisticRegression()
model.fit(x, result)

# Get news from user
user_news = input("Enter news: ")

# Convert user news into numbers
user_data = vectorizer.transform([user_news])

# Prediction
prediction = model.predict(user_data)

if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")
