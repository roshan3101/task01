# Smart Product Recommendation System

## Overview
This sophisticated recommendation system provides personalized product suggestions using advanced machine learning algorithms. Built with Flask and Python, the system leverages multiple recommendation strategies to deliver highly relevant product suggestions based on user preferences and behavior.

## Key Features

### Multiple Recommendation Approaches
- **Hybrid Recommendations**: Combines multiple algorithms with weighted scoring for optimal results
- **Collaborative Filtering**: Identifies similar users based on both interaction patterns and interest tags
- **Content-Based Filtering**: Analyzes product attributes using TF-IDF vectorization
- **Basic Popularity-Based**: Fallback strategy for new users leveraging category preferences

### Interactive User Experience
- **Real-time Updates**: Recommendations refresh instantly when users like products
- **Dynamic Algorithm Selection**: Users can switch between different recommendation approaches
- **Responsive Design**: Modern UI that works across all device sizes
- **Visual Category Organization**: Products are organized by category preference

### Analytics and Insights
- **System Metrics Dashboard**: Track total users, products, and interactions
- **Popular Products**: Identify trending products based on user likes
- **User Behavior Analysis**: Understand user preferences and interaction patterns

## Technical Implementation

### Architecture
- **Backend**: Flask web application with Python 3.9+
- **Recommendation Engine**: Scikit-learn based algorithms with custom weighting
- **Data Storage**: JSON-based interaction tracking with CSV data sources
- **Frontend**: HTML/CSS/JavaScript with dynamic content rendering

### Algorithm Details

#### Hybrid Recommendation Engine
The system combines three different recommendation approaches with weighted scoring:
- Collaborative Filtering (60% weight): Uses both interaction patterns and interest similarities
- Content-Based Filtering (30% weight): TF-IDF based product similarity
- Basic Recommendations (10% weight): Category and popularity based

#### Collaborative Filtering Enhancements
- User similarity calculation based on both product interactions (70%) and interests (30%)
- Jaccard similarity for interest tag matching
- Weighted product scoring based on the similarity of recommending users

#### Content-Based Filtering
- TF-IDF vectorization of product attributes (categories and tags)
- Pre-computed product similarity matrix
- Aggregated similarity scores based on user's interaction history

### Data Processing
- Text preprocessing with regular expressions for TF-IDF
- Feature extraction from user and product attributes
- Real-time recommendation matrix updates

## Installation and Usage

### Prerequisites
- Python 3.9+
- Conda or virtual environment

### Setup
1. Clone the repository
2. Create a conda environment:
```bash
conda create -n recommendation_system python=3.9
conda activate recommendation_system
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Access the application at http://localhost:5000

### Usage Guide
1. Select a user from the homepage
2. Browse personalized recommendations
3. Click the "Like" button on products of interest
4. Watch as recommendations dynamically update
5. Try different recommendation approaches using the buttons
6. View system analytics via the "Analytics" button


## Content Filtering Logic

Our content-based filtering analyzes product attributes using TF-IDF vectorization:

1. **Feature Extraction**: Product features (category and tags) are combined into a single content feature:
   ```python
   self.products['content_features'] = self.products['category'] + ' ' + self.products['tags']
   ```

2. **TF-IDF Vectorization**: Text features are converted to numerical vectors that capture term importance:
   ```python
   self.tfidf = TfidfVectorizer(stop_words='english')
   self.tfidf_matrix = self.tfidf.fit_transform(self.products['content_features'])
   ```

3. **Similarity Calculation**: A product similarity matrix is pre-computed using cosine similarity:
   ```python
   self.product_similarity = cosine_similarity(self.tfidf_matrix)
   ```

4. **Recommendation Generation**: When a user has liked products, we average the similarity scores from those products to find new recommendations:
   ```python
   similarity_scores = np.zeros(len(self.products))
   for idx in product_indices:  # indices of products the user liked
       similarity_scores += self.product_similarity[idx]
   similarity_scores = similarity_scores / len(product_indices)
   ```

## Collaborative Filtering Simulation

Our collaborative filtering identifies similar users and their preferences:

1. **User-Product Interaction Matrix**: We build a binary matrix where rows represent users and columns represent products:
   ```python
   interaction_matrix = np.zeros((len(all_users), len(all_products)))
   for i, user in enumerate(all_users):
       for j, product in enumerate(all_products):
           if product in self.interactions[user]:
               interaction_matrix[i, j] = 1
   ```

2. **User Similarity**: We calculate similarity between users based on both:
   - **Product interactions** (70% weight): Using cosine similarity between interaction vectors
   - **Interest tags** (30% weight): Using Jaccard similarity on user interests
   ```python
   combined_similarity = 0.7 * product_similarity + 0.3 * np.array(interest_similarity)
   ```

3. **Weighted Recommendations**: Products are scored based on similar users' preferences, weighted by similarity:
   ```python
   for similar_user_id in similar_users:
       similarity_weight = combined_similarity[user_idx]
       for product in self.interactions[similar_user_id]:
           recommended_products[product] += similarity_weight
   ```

## User Profile Storage and Updates

User profiles are stored and updated through interaction tracking:

1. **Data Structure**: User interactions are stored in a dictionary mapping users to sets of liked product IDs:
   ```python
   self.interactions = {
       'user123': {101, 204, 356},  # Product IDs user 123 has liked
       'user456': {102, 204, 415}   # Product IDs user 456 has liked
   }
   ```

2. **Persistence**: Interactions are serialized to JSON for persistent storage:
   ```python
   def _save_interactions(self):
       # Convert sets to lists for JSON serialization
       serializable_interactions = {user_id: list(products) for user_id, products in self.interactions.items()}
       with open(Config.INTERACTIONS_FILE, 'w') as f:
           json.dump(serializable_interactions, f)
   ```

3. **Profile Updates**: When a user likes a product, their profile is immediately updated:
   ```python
   def add_interaction(self, user_id, product_id):
       if user_id not in self.interactions:
           self.interactions[user_id] = set()
       self.interactions[user_id].add(product_id)
       self._save_interactions()
   ```

4. **Preference Extraction**: User preferences are derived from both explicit data (CSV) and implicit interaction history. 
