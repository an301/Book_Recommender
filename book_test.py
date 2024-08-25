import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

book_data = pd.read_csv('COPY PATH')
book_data = book_data.reset_index()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#prints data
# print(book_data.head())


# number of rows and columns in the data frame
# print(book_data.shape)

# selecting the relevant features for recommendation
selected_features = ['book_title','book_details','publication_info','author','num_pages']
# print(selected_features)

# replacing the null valuess with null string
for feature in selected_features:
  book_data[feature] = book_data[feature].fillna('')

# combining all the 5 selected features
combined_features = book_data['book_title']+' '+book_data['book_details']+' '+book_data['publication_info']+' '+book_data['author']+' '+book_data['num_pages']
# print(combined_features)
#
# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)


# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
# print(similarity)
# print(similarity.shape)


# getting the book name from the user
book_name = input(' Enter your favorite book: ')

# creating a list with all the book names given in the dataset
list_of_all_titles = book_data['book_title'].tolist()
# print(list_of_all_titles)

# finding the close match for the book name given by the user
find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
# print(find_close_match)

close_match = find_close_match[0]
# print(close_match)

# finding the index of the book with title
index_of_the_book = book_data[book_data.book_title == close_match]['index'].values[0]
# print(index_of_the_book)

# getting a list of similar books
similarity_score = list(enumerate(similarity[index_of_the_book]))
# print(similarity_score)
# print(len(similarity_score))  #4083 values (getting similarity scores of all other books)

# sorting the books based on their similarity score
sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True)
# print(sorted_similar_books)

# print the name of similar books based on the index
print('Books suggested for you : \n')

i = 1

for book in sorted_similar_books:
  index = book[0]
  title_from_index = book_data[book_data.index==index]['book_title'].values[0]
  publication_info_from_index = book_data[book_data.index==index]['publication_info'].values[0]  # Get the publication info
  if i < 31:
    print(f"{i}. {title_from_index} ({publication_info_from_index})")  # Print title and publication info
    i += 1
