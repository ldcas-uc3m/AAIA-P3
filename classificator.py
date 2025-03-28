from openai import OpenAI
import json
import os
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize OpenAI client that points to the local LM Studio server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def classify_text(text):
    """Classify a single text using the model"""
    messages = [
        {"role": "system", "content": "You are a text classifier. Given a news article, classify it into one of the following categories: business, entertainment, politics, sports, tech. You must return a json containing the classification. "},
        {"role": "user", "content": text}
    ]

    # Define the expected response structure
    character_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "characters",
            "schema": {
                "type": "object",
                "properties": {
                    "classification": {
                    "type": "string",
                    "description": "The classification of the item",
                    "enum": [
                        "business",
                        "entertainment",
                        "politics",
                        "sports",
                        "tech"
                    ]
                    }
                },
                "required": [
                    "classification"
                ],
            },
        }
    }

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b-instruct-mlx@4bit",
            messages=messages,
            response_format=character_schema,
            temperature=0  # Set temperature to 0 for deterministic outputs
        )
        results = json.loads(response.choices[0].message.content)
        return results["classification"]
    except Exception as e:
        print(f"Error classifying text: {e}")
        return None

def read_articles(base_folder):
    """Read all articles from the folder structure and return them with their true categories"""
    articles = []
    true_labels = []
    file_paths = []
    
    categories = ["business", "entertainment", "politics", "sports", "tech"]
    
    for category in categories:
        category_path = Path(base_folder) / category
        if not category_path.exists():
            print(f"Warning: Category folder {category_path} not found")
            continue
            
        for file_path in category_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                articles.append(content)
                true_labels.append(category)
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return articles, true_labels, file_paths

def evaluate_classifier(base_folder):
    """Evaluate the classifier on all articles in the folder structure"""
    start_time = time.time()  # Start timing execution
    
    print(f"Reading articles from {base_folder}...")
    articles, true_labels, file_paths = read_articles(base_folder)
    
    if not articles:
        print("No articles found to classify.")
        return
    
    print(f"Found {len(articles)} articles. Starting classification...")
    predictions = []
    
    for i, (article, true_label, file_path) in enumerate(zip(articles, true_labels, file_paths)):
        print(f"Classifying article {i+1}/{len(articles)} from {file_path.name}...")
        prediction = classify_text(article)
        predictions.append(prediction)
        
        # Print progress
        if prediction:
            print(f"  Predicted: {prediction}, Actual: {true_label}")
        else:
            print(f"  Failed to classify")
        
    
    # Filter out None predictions
    valid_predictions = [p for p in predictions if p is not None]
    valid_true_labels = [true_labels[i] for i, p in enumerate(predictions) if p is not None]
    
    if not valid_predictions:
        print("No valid predictions were made.")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    conf_matrix = confusion_matrix(valid_true_labels, valid_predictions, 
                                 labels=["business", "entertainment", "politics", "sports", "tech"])
    report = classification_report(valid_true_labels, valid_predictions)
    
    # Print results
    print("\n--- EVALUATION RESULTS ---")
    print(f"Total articles: {len(articles)}")
    print(f"Successfully classified articles: {len(valid_predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Get folder path from user
    evaluate_classifier("./bbc-test")