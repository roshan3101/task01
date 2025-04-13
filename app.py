from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from recommender import Recommender
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
recommender = Recommender()

@app.route('/')
def index():
    users = recommender.users['User_ID'].tolist()
    return render_template('index.html', users=users)

@app.route('/recommendations/')
def recommendations_redirect():
    # Redirect to home page when no user_id is provided
    return redirect(url_for('index'))

@app.route('/recommendations/<user_id>')
def recommendations(user_id):
    # Get recommendations
    hybrid_recs = recommender.get_hybrid_recommendations(user_id)
    all_products = recommender.get_all_products(user_id)
    
    # Get user's liked products
    user_likes = []
    if user_id in recommender.interactions:
        user_likes = list(recommender.interactions[user_id])
    
    return render_template('recommendations.html',
                         user_id=user_id,
                         recommended_products=hybrid_recs.to_dict('records'),
                         category_products=all_products['category_products'].to_dict('records'),
                         other_products=all_products['other_products'].to_dict('records'),
                         user_likes=user_likes)

@app.route('/recommendations/<user_id>/<approach>')
def recommendations_by_approach(user_id, approach):
    # Get recommendations based on approach
    if approach == 'hybrid':
        recommendations = recommender.get_hybrid_recommendations(user_id)
    elif approach == 'collaborative':
        recommendations = recommender.get_collaborative_recommendations(user_id)
    elif approach == 'content':
        recommendations = recommender.get_content_based_recommendations(user_id)
    else:  # 'basic'
        recommendations = recommender.get_basic_recommendations(user_id)
    
    # Get user's liked products
    user_likes = []
    if user_id in recommender.interactions:
        user_likes = list(recommender.interactions[user_id])
    
    return jsonify({
        'recommendations': recommendations.to_dict('records'),
        'user_likes': user_likes
    })

@app.route('/like_product', methods=['POST'])
def like_product():
    data = request.json
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    
    if user_id and product_id:
        recommender.add_interaction(user_id, int(product_id))
        
        # Get updated recommendations
        recommended_products = recommender.get_hybrid_recommendations(user_id)
        
        # Convert to dict for JSON response
        recommendations_data = recommended_products.to_dict('records')
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations_data
        })
    return jsonify({'status': 'error'}), 400

@app.route('/analytics')
def analytics():
    # Get analytics data
    total_users = len(recommender.users)
    total_products = len(recommender.products)
    total_interactions = sum(len(interactions) for interactions in recommender.interactions.values())
    
    # Get most popular products
    product_counts = {}
    for user_id, products in recommender.interactions.items():
        for product in products:
            if product not in product_counts:
                product_counts[product] = 0
            product_counts[product] += 1
    
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
    top_products = []
    for product_id, count in sorted_products[:10]:
        product_data = recommender.products[recommender.products['product_id'] == product_id]
        if not product_data.empty:
            top_products.append({
                'product_id': product_id,
                'title': product_data.iloc[0]['title'],
                'category': product_data.iloc[0]['category'],
                'likes': count
            })
    
    return render_template('analytics.html',
                         total_users=total_users,
                         total_products=total_products,
                         total_interactions=total_interactions,
                         top_products=top_products)

if __name__ == '__main__':
    app.run(debug=True) 