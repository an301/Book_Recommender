# Book Recommender System

## Project Overview

This is a **Book Recommender System** that suggests books based on specific features from a dataset. Users can input the title of a book, and the system will return a list of 30 similar books.

## Features Used for Recommendations

The system focuses on the following book attributes to generate recommendations:

- **Title**: The name of the book.
- **Authors**: The authors of the book.
- **Categories**: The genres or categories the book belongs to.
- **Description**: A brief summary or synopsis of the book.
- **Published Year**: The year the book was published.

## How It Works

1. **Input**: The user enters the title of a book they like.
2. **Processing**: The system matches the input with books in the dataset, comparing features like title, authors, and categories.
3. **Output**: A list of up to 30 similar books is displayed to the user. If fewer than 30 books are found, or the input title doesn’t exist, an appropriate message is shown.

## Dataset Information

The dataset includes various book attributes, such as:
- Title
- Authors
- Categories
- Thumbnail (cover image)
- Description
- Published Year

## Future Enhancements

- Improving recommendation accuracy by fine-tuning the feature weighting.
- Allowing users to search based on different criteria (e.g., authors, categories).

## How to Run

1. Open this website and enter a book: https://an301.github.io/Book_Recommender/
2. Click button to get recommendations
