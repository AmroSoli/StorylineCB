# process_course_content.py
import os
import json
import openai

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Directory containing text chunks
TEXT_CHUNKS_DIR = r'V:\AIChatBot\StorylineChatbot\white_card_course'

# Path to save embeddings
EMBEDDINGS_PATH = 'course_embeddings.json'

def load_text_chunks():
    text_chunks = []
    for filename in os.listdir(TEXT_CHUNKS_DIR):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(TEXT_CHUNKS_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text_chunks.append(text)
    return text_chunks

def generate_embeddings(text_chunks):
    embeddings = []
    for text in text_chunks:
        response = openai.Embedding.create(
            input=text,
            model='text-embedding-ada-002'
        )
        embedding = response['data'][0]['embedding']
        embeddings.append({'text': text, 'embedding': embedding})
    return embeddings

def process_course_content():
    text_chunks = load_text_chunks()
    print(f"Loaded {len(text_chunks)} text chunks.")
    embeddings = generate_embeddings(text_chunks)
    print("Generated embeddings for all text chunks.")
    # Save embeddings to JSON
    with open(EMBEDDINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}.")

if __name__ == '__main__':
    process_course_content()
