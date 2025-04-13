import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Get the base directory of the application
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')
    USERS_FILE = os.path.join(DATA_FOLDER, 'user_personalized_features.csv')
    PRODUCTS_FILE = os.path.join(DATA_FOLDER, 'products.csv')
    INTERACTIONS_FILE = os.path.join(DATA_FOLDER, 'user_interactions.json') 