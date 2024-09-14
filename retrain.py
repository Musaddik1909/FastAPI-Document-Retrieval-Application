# retrain.py

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import sqlite3
import logging

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database setup
db_connection = sqlite3.connect('documents.db', check_same_thread=False)
cursor = db_connection.cursor()

# Function to retrieve training data
def get_training_data():
    cursor.execute("SELECT text FROM documents")
    documents = cursor.fetchall()
    examples = []
    
    # Creating input examples for fine-tuning
    for doc in documents:
        text = doc[0]
        # Assume pairs of text are consecutive rows, fine-tune for semantic similarity
        examples.append(InputExample(texts=[text, text], label=1.0))  # Use label 1.0 for identical texts
    return examples

# Function to fine-tune the model
def fine_tune_model():
    # Retrieve training data
    train_examples = get_training_data()

    # DataLoader for training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    logging.info("Starting fine-tuning of the model.")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    logging.info("Fine-tuning completed.")

    # Save the fine-tuned model
    model.save('fine_tuned_model')

if __name__ == "__main__":
    logging.basicConfig(filename='retrain.log', level=logging.INFO)
    fine_tune_model()
