from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
# filter out pyarrow requirement warning (pyarrow is not ported to pyscript)
warnings.filterwarnings('ignore')

from pyweb import pydom
from pyscript import document
from pyscript import display
import pandas as pd
import difflib
from pyodide.http import open_url

text_box_content = pydom["#book_name"]

recommendation_template = pydom.Element(document.querySelector("#recommendation-template").content.querySelector(".query-main"))

# Load and preprocess the dataset
url_content = open_url("https://raw.githubusercontent.com/an301/Book_Recommender/main/data.csv")
book_data = pd.read_csv(url_content)
book_data = book_data.reset_index()

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

def button_pressed(event):
    if not text_box_content.value[0]:
        pydom["div#result_message"].html = "<h2>Please enter a value.</h2>"
        pydom["div#recommendations-list"].html = ""
        return None

    index = selected_book_index()

    if index == None:
        pydom["div#result_message"].html = "<h2>Match not found.</h2>"
        pydom["div#recommendations-list"].html = ""
        return None
    
    
    pydom["div#result_message"].html = "<h2>Books suggested for you:</h2>"
    pydom["div#recommendations-list"].html = ""

    get_top_books(index)

def selected_book_index():
    book_name = text_box_content.value[0]
    list_of_all_titles = book_data['title'].tolist()

    # Find the closest match for the input book name
    find_close_match = difflib.get_close_matches(book_name.lower(), [title.lower() for title in list_of_all_titles])

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_book = book_data[book_data.title.str.lower() == close_match].index[0]
        return index_of_the_book
    else:
        return None

def get_top_books(index_of_the_book):

    # Calculate cosine similarity
    selected_book_vector = feature_vectors[index_of_the_book]
    similarity_score = cosine_similarity(selected_book_vector, feature_vectors).flatten()

    # Sort books based on similarity scores
    sorted_similar_books = sorted(list(enumerate(similarity_score)), key=lambda x: x[1], reverse=True)
    # Collect the top 30 recommendations
    for i, book in enumerate(sorted_similar_books[1:31], start=1):  # start from 1 to exclude the input book
        index = book[0]
        display_recommended_book(index)

def display_recommended_book(index):
    book_info = book_data.loc[index]

    recommendation_html = recommendation_template.clone()
    recommendation_html.id = "book-index-" + str(index)

    recommendation_html._js.querySelector(".query-title").textContent = book_info["title"]
    recommendation_html._js.querySelector(".query-author").textContent = book_info["authors"]
    recommendation_html._js.querySelector(".query-published").textContent = book_info["published_year"]
    recommendation_html._js.querySelector(".query-image").src = book_info["thumbnail"]

    pydom["#recommendations-list"][0].append(recommendation_html)