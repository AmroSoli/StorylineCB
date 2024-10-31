# process_course_content.py

import os
import json
import numpy as np
from openai import OpenAI

# Instantiate the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embeddings(text_chunks):
    embeddings = []
    for text in text_chunks:
        response = client.embeddings.create(
            input=text,
            model='text-embedding-ada-002'
        )
        embedding = response.data[0].embedding
        embeddings.append({'text': text, 'embedding': embedding})
    return embeddings

def process_course_content():
    # Load your course content and split it into chunks
    with open('course_content.txt', 'r', encoding='utf-8') as f:
        course_content = f.read()
    # Split course_content into chunks (e.g., paragraphs)
    text_chunks = course_content.split('\n\n')
    print(f"Loaded {len(text_chunks)} text chunks.")

    # Generate embeddings
    embeddings = generate_embeddings(text_chunks)

    # Save embeddings to a JSON file
    with open('course_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False)
    print("Saved embeddings to course_embeddings.json.")

if __name__ == '__main__':
    process_course_content()
