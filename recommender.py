import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import re
from config import Config

class Recommender:
    def __init__(self):
        self.users = pd.read_csv(Config.USERS_FILE)
        self.products = pd.read_csv(Config.PRODUCTS_FILE)
        self.interactions = self._load_interactions()
        # Initialize content-based features
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data for content-based filtering"""
        # Create content features from product attributes
        self.products['content_features'] = self.products['category'] + ' ' + self.products['tags']
        
        # Clean the text features
        self.products['content_features'] = self.products['content_features'].apply(lambda x: 
            ' '.join(re.sub(r'[^\w\s]', ' ', str(x).lower()).split()))
        
        # Create TF-IDF vectors for content features
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.products['content_features'])
        
        # Pre-compute product similarity matrix
        self.product_similarity = cosine_similarity(self.tfidf_matrix)
        
    def _load_interactions(self):
        if os.path.exists(Config.INTERACTIONS_FILE):
            with open(Config.INTERACTIONS_FILE, 'r') as f:
                interactions_dict = json.load(f)
                # Convert lists back to sets
                return {user_id: set(products) for user_id, products in interactions_dict.items()}
        return {}
    
    def _save_interactions(self):
        # Convert sets to lists for JSON serialization
        serializable_interactions = {user_id: list(products) for user_id, products in self.interactions.items()}
        with open(Config.INTERACTIONS_FILE, 'w') as f:
            json.dump(serializable_interactions, f)
    
    def get_user_preferences(self, user_id):
        """Get user's category preference and interests"""
        user_data = self.users[self.users['User_ID'] == user_id].iloc[0]
        return {
            'category': user_data['Product_Category_Preference'],
            'interests': user_data['Interests']
        }
    
    def get_basic_recommendations(self, user_id, top_n=5):
        """Get recommendations based on user's category preference and product popularity"""
        preferences = self.get_user_preferences(user_id)
        category_products = self.products[self.products['category'] == preferences['category']]
        
        # Sort by popularity score and get top N
        return category_products.sort_values('popularity_score', ascending=False).head(top_n)
    
    def get_content_based_recommendations(self, user_id, top_n=5):
        """Get content-based recommendations based on user's past interactions"""
        if user_id not in self.interactions or not self.interactions[user_id]:
            return self.get_basic_recommendations(user_id, top_n)
        
        # Get user's interaction history
        user_products = list(self.interactions[user_id])
        
        # Find product indexes in the dataframe
        product_indices = [self.products[self.products['product_id'] == pid].index[0] 
                          for pid in user_products if pid in self.products['product_id'].values]
        
        if not product_indices:
            return self.get_basic_recommendations(user_id, top_n)
        
        # Calculate average similarity scores for all products
        similarity_scores = np.zeros(len(self.products))
        for idx in product_indices:
            similarity_scores += self.product_similarity[idx]
        
        # Average the scores and get top n products
        similarity_scores = similarity_scores / len(product_indices)
        
        # Get top product indices excluding products user already interacted with
        recommended_indices = []
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        for idx in sorted_indices:
            product_id = self.products.iloc[idx]['product_id']
            if product_id not in user_products:
                recommended_indices.append(idx)
                if len(recommended_indices) >= top_n:
                    break
        
        return self.products.iloc[recommended_indices]
    
    def get_collaborative_recommendations(self, user_id, top_n=5):
        """Get recommendations based on similar users' preferences and interests"""
        if user_id not in self.interactions:
            return self.get_basic_recommendations(user_id, top_n)
            
        # Create user-product interaction matrix
        all_users = list(self.interactions.keys())
        all_products = self.products['product_id'].tolist()
        
        # If no other users with interactions, return basic recommendations
        if len(all_users) <= 1:
            return self.get_basic_recommendations(user_id, top_n)
        
        # Create interaction matrix
        interaction_matrix = np.zeros((len(all_users), len(all_products)))
        for i, user in enumerate(all_users):
            for j, product in enumerate(all_products):
                if product in self.interactions[user]:
                    interaction_matrix[i, j] = 1
        
        # Create interest similarity matrix
        target_user_data = self.users[self.users['User_ID'] == user_id]
        if len(target_user_data) == 0:
            return self.get_basic_recommendations(user_id, top_n)
            
        target_interests = target_user_data.iloc[0]['Interests'].split(',')
        interest_similarity = []
        
        for u in all_users:
            user_data = self.users[self.users['User_ID'] == u]
            if len(user_data) == 0:
                interest_similarity.append(0)
                continue
                
            user_interests = user_data.iloc[0]['Interests'].split(',')
            
            # Calculate Jaccard similarity for interests
            intersection = len(set(target_interests) & set(user_interests))
            union = len(set(target_interests) | set(user_interests))
            similarity = intersection / union if union > 0 else 0
            interest_similarity.append(similarity)
        
        # Find similar users based on product interactions
        target_user_idx = all_users.index(user_id)
        user_vector = interaction_matrix[target_user_idx].reshape(1, -1)
        product_similarity = cosine_similarity(user_vector, interaction_matrix)[0]
        
        # Combine product and interest similarity (weighted average)
        combined_similarity = 0.7 * product_similarity + 0.3 * np.array(interest_similarity)
        
        # Get top similar users
        similar_users = np.argsort(combined_similarity)[::-1][1:6]  # Exclude the user themselves, get top 5
        
        # Get products liked by similar users, weighted by similarity
        recommended_products = {}
        for i, user_idx in enumerate(similar_users):
            if combined_similarity[user_idx] <= 0:
                continue
                
            similar_user_id = all_users[user_idx]
            similarity_weight = combined_similarity[user_idx]
            
            for product in self.interactions[similar_user_id]:
                if product not in recommended_products:
                    recommended_products[product] = 0
                recommended_products[product] += similarity_weight
        
        # Remove products already liked by the user
        for product in self.interactions[user_id]:
            if product in recommended_products:
                del recommended_products[product]
        
        # If no recommended products after filtering, fall back to basic recommendations
        if not recommended_products:
            return self.get_content_based_recommendations(user_id, top_n)
        
        # Sort products by weighted score
        sorted_products = sorted(recommended_products.items(), key=lambda x: x[1], reverse=True)
        top_product_ids = [p[0] for p in sorted_products[:top_n]]
        
        # Get product details
        collaborative_recs = self.products[self.products['product_id'].isin(top_product_ids)]
        
        # Preserve the order of sorted recommendations
        if not collaborative_recs.empty:
            collaborative_recs = collaborative_recs.set_index('product_id').loc[top_product_ids].reset_index()
        
        # If we don't have enough collaborative recommendations, supplement with content-based ones
        if len(collaborative_recs) < top_n:
            content_recs = self.get_content_based_recommendations(user_id, top_n)
            # Filter out products already in collaborative recommendations
            if not collaborative_recs.empty:
                content_recs = content_recs[~content_recs['product_id'].isin(collaborative_recs['product_id'])]
            # Combine and take top N
            combined = pd.concat([collaborative_recs, content_recs]).head(top_n)
            return combined
            
        return collaborative_recs
    
    def get_hybrid_recommendations(self, user_id, top_n=5):
        """Get hybrid recommendations combining collaborative, content-based and basic approaches"""
        # Get recommendations from different approaches
        collaborative_recs = self.get_collaborative_recommendations(user_id, top_n)
        content_recs = self.get_content_based_recommendations(user_id, top_n)
        basic_recs = self.get_basic_recommendations(user_id, top_n)
        
        # Assign weights to each approach
        collaborative_weight = 0.6
        content_weight = 0.3
        basic_weight = 0.1
        
        # Combine all product IDs
        all_product_ids = set()
        products_scores = {}
        
        # Process collaborative recommendations
        for _, row in collaborative_recs.iterrows():
            product_id = row['product_id']
            all_product_ids.add(product_id)
            if product_id not in products_scores:
                products_scores[product_id] = 0
            # Assign score based on position in list and weight
            position = list(collaborative_recs['product_id']).index(product_id)
            score = (top_n - position) / top_n * collaborative_weight
            products_scores[product_id] += score
        
        # Process content-based recommendations
        for _, row in content_recs.iterrows():
            product_id = row['product_id']
            all_product_ids.add(product_id)
            if product_id not in products_scores:
                products_scores[product_id] = 0
            # Assign score based on position in list and weight
            position = list(content_recs['product_id']).index(product_id)
            score = (top_n - position) / top_n * content_weight
            products_scores[product_id] += score
        
        # Process basic recommendations
        for _, row in basic_recs.iterrows():
            product_id = row['product_id']
            all_product_ids.add(product_id)
            if product_id not in products_scores:
                products_scores[product_id] = 0
            # Assign score based on position in list and weight
            position = list(basic_recs['product_id']).index(product_id)
            score = (top_n - position) / top_n * basic_weight
            products_scores[product_id] += score
            
        # Sort by score and get top products
        sorted_products = sorted(products_scores.items(), key=lambda x: x[1], reverse=True)
        top_product_ids = [p[0] for p in sorted_products[:top_n]]
        
        # Filter products based on top IDs
        hybrid_recs = self.products[self.products['product_id'].isin(top_product_ids)]
        
        # Preserve the order of sorted recommendations
        hybrid_recs = hybrid_recs.set_index('product_id').loc[top_product_ids].reset_index()
        
        return hybrid_recs
    
    def add_interaction(self, user_id, product_id):
        """Add a new user-product interaction"""
        if user_id not in self.interactions:
            self.interactions[user_id] = set()
        self.interactions[user_id].add(product_id)
        self._save_interactions()
    
    def get_all_products(self, user_id, top_n=10):
        """Get all products with user's category preference prioritized"""
        preferences = self.get_user_preferences(user_id)
        category_products = self.products[self.products['category'] == preferences['category']]
        other_products = self.products[self.products['category'] != preferences['category']]
        
        # Sort by popularity
        category_products = category_products.sort_values('popularity_score', ascending=False)
        other_products = other_products.sort_values('popularity_score', ascending=False)
        
        return {
            'category_products': category_products.head(top_n),
            'other_products': other_products.head(top_n)
        } 