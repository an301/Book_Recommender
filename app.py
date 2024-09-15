from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the dataset
book_data = pd.read_csv('put location of file here')
book_data = book_data.reset_index()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Select relevant features and handle missing values
selected_features = ['title', 'authors', 'categories', 'description', 'published_year']
for feature in selected_features:
    book_data[feature] = book_data[feature].fillna('')

# Convert 'published_year' to integer
book_data['published_year'] = pd.to_numeric(book_data['published_year'], errors='coerce').fillna(0).astype(int)

# Feature weighting
weight_authors = 2
weight_categories = 10
weight_description = 15

combined_features = (
    book_data['title'] + ' ' +
    weight_authors * book_data['authors'] + ' ' +
    weight_categories * book_data['categories'] + ' ' +
    weight_description * book_data['description']
)

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
feature_vectors = vectorizer.fit_transform(combined_features)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    message = ''
    if request.method == 'POST':
        book_name = request.form['book_name']
        recommendations, message = get_recommendations(book_name)

    return render_template('index.html', recommendations=recommendations, message=message)

def get_recommendations(book_name):
    recommendations = []
    message = ''
    list_of_all_titles = book_data['title'].tolist()

    # Find the closest match for the input book name
    find_close_match = difflib.get_close_matches(book_name.lower(), [title.lower() for title in list_of_all_titles])
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_book = book_data[book_data.title.str.lower() == close_match].index[0]

        # Calculate cosine similarity
        selected_book_vector = feature_vectors[index_of_the_book]
        similarity_score = cosine_similarity(selected_book_vector, feature_vectors).flatten()

        # Sort books based on similarity scores
        sorted_similar_books = sorted(list(enumerate(similarity_score)), key=lambda x: x[1], reverse=True)

        # Collect the top 30 recommendations
        for i, book in enumerate(sorted_similar_books[1:31], start=1):  # start from 1 to exclude the input book
            index = book[0]
            title_from_index = book_data.iloc[index]['title']
            thumbnail_from_index = book_data.iloc[index]['thumbnail']
            published_year_from_index = book_data.iloc[index]['published_year']

            recommendations.append({
                'title': title_from_index,
                'thumbnail': thumbnail_from_index,
                'published_year': published_year_from_index
            })

        # Handle case where fewer than 30 recommendations are found
        if len(recommendations) < 30:
            message = f"Only {len(recommendations)} recommendations were found."
    else:
        message = "Book not found in the dataset. Please try another title."

    return recommendations, message

if __name__ == '__main__':
    app.run(debug=True)
