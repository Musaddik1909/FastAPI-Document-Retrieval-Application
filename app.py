from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
import time
import logging
from threading import Thread
import requests
import random
import redis
import json

# Initialize FastAPI and logging
app = FastAPI()
logging.basicConfig(filename='api.log', level=logging.INFO)

# Initialize Redis client (use the container name as the host)
try:
    redis_client = redis.Redis(host='redis-container', port=6379, db=0, decode_responses=True)
    logging.info("Connected to Redis.")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {str(e)}")
    raise HTTPException(status_code=500, detail="Redis connection failed")

# Initialize SentenceTransformer model for semantic search
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# In-memory cache for search results (using Redis)
def cache_get(key):
    try:
        return json.loads(redis_client.get(key))
    except Exception as e:
        logging.error(f"Error fetching from cache: {str(e)}")
        return None

def cache_set(key, value):
    try:
        redis_client.set(key, json.dumps(value))
    except Exception as e:
        logging.error(f"Error setting cache: {str(e)}")

# Background Scraper Function
def background_scraper():
    while True:
        try:
            # Fetch the IDs of the top stories from Hacker News
            top_stories_response = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty")
            top_stories_ids = top_stories_response.json()[:10]  # Get the top 10 story IDs

            for story_id in top_stories_ids:
                # Fetch the details for each story
                story_response = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json?print=pretty")
                story_data = story_response.json()

                # Extract the title and URL of the story
                title = story_data.get('title', '')
                url = story_data.get('url', '')
                if not url:  # Skip items without URLs
                    continue

                # Combine title and URL for storage
                text = f"{title} - {url}"

                # Store the document in Redis with a unique key
                redis_client.lpush('documents', text)

            time.sleep(3600)  # Scrape every hour

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error during scraping: {str(e)}")
        except Exception as e:
            logging.error(f"Error during scraping: {str(e)}")

# Start the scraper in a separate thread
Thread(target=background_scraper).start()

# Helper function to get or create a user
def get_or_create_user(user_id):
    try:
        # Retrieve the user from Redis
        user = redis_client.hgetall(f"user:{user_id}")
        if user:
            request_count = int(user.get('request_count', 0)) + 1
            redis_client.hmset(f"user:{user_id}", {"request_count": request_count, "last_request_time": time.time()})
        else:
            request_count = 1
            redis_client.hmset(f"user:{user_id}", {"request_count": request_count, "last_request_time": time.time()})
        return request_count
    except Exception as e:
        logging.error(f"Error in user retrieval or creation: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis error")

# Request and Response Models
class SearchRequest(BaseModel):
    user_id: str
    text: str
    top_k: int = 5
    threshold: float = 0.5

class SearchResult(BaseModel):
    results: List[str]

# New route to handle requests at the root URL
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI Document Retrieval Application with Redis!"}

# Endpoint to check API health
@app.get("/health")
async def health_check():
    random_response = random.choice(["API is active", "Server is up", "Running smoothly"])
    return {"status": random_response}

# Endpoint to handle search requests
@app.post("/search", response_model=SearchResult)
async def search(request: SearchRequest):
    try:
        start_time = time.time()
        user_id = request.user_id
        request_count = get_or_create_user(user_id)

        # Rate limiting: Max 5 requests per user
        if request_count > 5:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests, please try again later.")

        # Check if response is cached in Redis
        cache_key = f"{user_id}:{request.text}"
        cached_result = cache_get(cache_key)
        if cached_result:
            logging.info(f"Cache hit for user {user_id}")
            return {"results": cached_result}

        # Retrieve documents from Redis
        documents = redis_client.lrange('documents', 0, -1)

        if not documents:
            logging.warning("No documents found in Redis.")
            raise HTTPException(status_code=404, detail="No documents found")

        # Re-Ranking using Sentence-Transformers
        query_embedding = model.encode(request.text, convert_to_tensor=True)
        doc_embeddings = model.encode(documents, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0].cpu().tolist()
        
        # Filter and sort results based on similarity score
        ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        ranked_results = [text for text, score in ranked_results if score > request.threshold][:request.top_k]

        # Cache the response in Redis
        cache_set(cache_key, ranked_results)

        # Logging response time
        response_time = time.time() - start_time
        logging.info(f"User {user_id} - Query: {request.text} - Response Time: {response_time} seconds")

        return {"results": ranked_results}

    except HTTPException as e:
        logging.error(f"HTTP error occurred: {str(e.detail)}")
        raise e
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # Run the Uvicorn server on all network interfaces (0.0.0.0) and port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
