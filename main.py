import pickle
import sys

def sentiment(reviews):
    vectorizer = pickle.load(open('vect_sent.pickle', 'rb'))
    reviews_transformed = vectorizer.transform(reviews).toarray()
    
    model = pickle.load(open('sentiment_classifier.pickle', 'rb'))
    predictions = model.predict(reviews_transformed)
    predictions = ['negative' if i==0 else 'neutral' if i==1 else 'positive' for i in predictions]
    return predictions

def category(reviews):
    vectorizer = pickle.load(open('vect_cat.pickle', 'rb'))
    reviews_transformed = vectorizer.transform(reviews).toarray()

    model = pickle.load(open('category_classifier.pickle', 'rb'))
    predictions = model.predict(reviews_transformed)
    predictions = ['noise' if i==0 else 'requirement' if i==1 else 'bug report' if i==2 else 'other' for i in predictions]
    return predictions

def predict(reviews):
    if sys.argv[1] == 'sentiment':
        predictions = sentiment(reviews)
    elif sys.argv[1] == 'category':
        predictions = category(reviews)
    for i, prediction in enumerate(predictions):
        print('\nreview {}: {}'.format(i, prediction))

def load_reviews():
    reviews = []
    count = 0
    while True:
        user_input = input('Enter review {}: '.format(count))
        if user_input == 'end':
            break
        reviews.append(user_input)
        count += 1
    if not reviews:
        quit()
    return reviews

def check_command():
    if len(sys.argv) != 2:
        return False
    return True

if check_command() is False:
    print("Please choose sentiment or category review analysis")
    quit()

reviews = load_reviews()
predict(reviews)



