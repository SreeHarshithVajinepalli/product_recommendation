from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the CSV file as a DataFrame
df = pd.read_csv('sample.csv')
# product_df = pd.read_csv('sample.csv')

def collaborative_filtering_recommendation_ratings(user_id_encoded):
    # Your existing collaborative filtering code here...
    # Preprocess the data to handle duplicate ratings by averaging
    df_preprocessed = df.groupby(['user_id_encoded', 'product_name'], as_index=False)['rating'].mean()

    # Create a user-item matrix
    user_item_matrix = df_preprocessed.pivot(index='user_id_encoded', columns='product_name', values='rating').fillna(0)

    # Calculate cosine similarity between users based on their ratings
    user_similarity = cosine_similarity(user_item_matrix)

    # Get the index of the input user
    user_index = user_id_encoded

    # Find users similar to the input user
    similar_users = np.argsort(user_similarity[user_index])[::-1]  # Most similar users first

    # Get the products highly rated by similar users but not by the input user
    input_user_rated = df_preprocessed[df_preprocessed['user_id_encoded'] == user_index]['product_name']
    recommended_products = set()

    for similar_user in similar_users:
        if similar_user == user_index:
            continue  # Skip the input user

        similar_user_rated = df_preprocessed[df_preprocessed['user_id_encoded'] == similar_user]['product_name']
        new_products = set(similar_user_rated) - set(input_user_rated)
        recommended_products.update(new_products)

        if len(recommended_products) >= 5:
            break

    # Convert recommended products set to a list
    recommended_products_list = list(recommended_products)[:5]

    # Create a DataFrame with the recommended products
    results_df = pd.DataFrame({
        'Id Encoded': [user_id_encoded] * len(recommended_products_list),
        'recommended product': recommended_products_list
    })
    return results_df

def recommend_products(user_id_encoded):
    # Use TfidfVectorizer to transform the product descriptions into numerical feature vectors
    tfidf = TfidfVectorizer(stop_words='english')
    df['about_product'] = df['about_product'].fillna('')  # fill NaN values with empty string
    tfidf_matrix = tfidf.fit_transform(df['about_product'])

    # Get the purchase history for the user
    user_history = df[df['user_id_encoded'] == user_id_encoded]

    # Use cosine_similarity to calculate the similarity between each pair of product descriptions
    # only for the products that the user has already purchased
    indices = user_history.index.tolist()

    if indices:
        # Create a new similarity matrix with only the rows and columns for the purchased products
        cosine_sim_user = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)

        # Create a pandas Series with product indices as the index and product names as the values
        products = df.iloc[indices]['product_name']
        indices = pd.Series(products.index, index=products)

        # Get the indices and similarity scores of products similar to the ones the user has already purchased
        similarity_scores = list(enumerate(cosine_sim_user[-1]))
        similarity_scores = [(i, score) for (i, score) in similarity_scores if i not in indices]

        # Sort the similarity scores in descending order
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 5 most similar products
        top_products = [i[0] for i in similarity_scores[1:6]]

        # Get the names of the top 5 most similar products
        recommended_products = df.iloc[top_products]['product_name'].tolist()

        # Get the reasons for the recommendation
        score = [similarity_scores[i][1] for i in range(5)]

        recommended_actual_prices = df.iloc[top_products]['actual_price'].tolist()
        recommended_ratings = df.iloc[top_products]['rating'].tolist()
        # Create a DataFrame with the results
        results_df = pd.DataFrame({'Id Encoded': [user_id_encoded] * 5,
                                   'recommended product': recommended_products,
                                   'score recommendation': score,
                                   'actual_price': recommended_actual_prices,
                                    'rating': recommended_ratings})
        print(results_df)
        return results_df
p_df = pd.read_csv("collaborative_purchase_recommendations_with_ratings_and_images.csv")
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        user_id_encoded = int(request.form['user_id_encoded'])

        collaborative_results = collaborative_filtering_recommendation_ratings(user_id_encoded)

        recommended_products_with_images = []
        for index, row in collaborative_results.iterrows():

            matching_p_rows = p_df[p_df['Id Encoded'] == row['Id Encoded']]
            for _, p_row in matching_p_rows.iterrows():
                recommended_products_with_images.append({
                    'Id Encoded': p_row['Id Encoded'],
                    'recommended product': p_row['recommended product'],
                    'img_link': p_row['img_link'],
                    'actual_price': p_row['product_price'],
                    'rating':p_row['product_rating']
                })
            break;
        
        description_results = recommend_products(user_id_encoded)
        recommended_products_with_images_ = []
        for index, row in description_results.iterrows():
            product_name = row['recommended product']
            matching_product = df[df['product_name'] == product_name]

            if not matching_product.empty:
                img_link = matching_product['img_link'].iloc[0]
                recommended_products_with_images_.append({
                    'Id Encoded': row['Id Encoded'],
                    'recommended product': product_name,
                    'img_link': img_link,
                    'actual_price':row['actual_price'],
                    'rating': row['rating']
                })
        
        return render_template('index.html', collaborative_results=recommended_products_with_images,
                               description_results=recommended_products_with_images_)
    return render_template('index.html', collaborative_results=None, description_results=None)

if __name__ == '__main__':
    app.run(debug=True)
