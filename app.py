import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Load and preprocess the dataset
book_data = pd.read_csv('')
book_data = book_data.reset_index()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Select relevant features and handle missing values
selected_features = ['book_title', 'book_details', 'publication_info', 'author', 'genres']

for feature in selected_features:
    book_data[feature] = book_data[feature].fillna('')

# Combine selected features into a single string for each book
combined_features = book_data['book_title'] + ' ' + book_data['book_details'] + ' ' + book_data['publication_info'] + ' ' + book_data['author'] + ' ' + book_data['genres']

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    message = ''
    if request.method == 'POST':
        book_name = request.form['book_name']
        list_of_all_titles = book_data['book_title'].tolist()

        # Find the closest match for the input book name
        find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
        if find_close_match:
            close_match = find_close_match[0]
            index_of_the_book = book_data[book_data.book_title == close_match].index[0]

            # calculate cosine similarity only for the selected book
            selected_book_vector = feature_vectors[index_of_the_book]
            similarity_score = cosine_similarity(selected_book_vector, feature_vectors).flatten()

            # Sort books based on similarity scores
            sorted_similar_books = sorted(list(enumerate(similarity_score)), key=lambda x: x[1], reverse=True)

            # Collect the top 30 recommendations
            for i, book in enumerate(sorted_similar_books[1:31], start=1):  # start from 1 to exclude the input book
                index = book[0]
                title_from_index = book_data.iloc[index]['book_title']
                author_from_index = book_data.iloc[index]['author']
                publication_info_from_index = book_data.iloc[index]['publication_info'].strip("[]")
                cover_image_uri = book_data.iloc[index]['cover_image_uri']

                recommendations.append({
                    'title': title_from_index,
                    'author': author_from_index,
                    'publication_info': publication_info_from_index,
                    'cover_image_uri': cover_image_uri
                })

            # Handle case where fewer than 30 recommendations are found
            if len(recommendations) < 30:
                message = f"Only {len(recommendations)} recommendations were found."
        else:
            message = "Book not found in the dataset. Please try another title."

    return render_template('index.html', recommendations=recommendations, message=message)

if __name__ == '__main__':
    app.run(debug=True)
