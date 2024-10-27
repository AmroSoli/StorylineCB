# process_pdfs.py
import os
import pickle
import numpy as np
import faiss
import PyPDF2
import nltk
import openai

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Directory containing PDFs
PDF_DIRECTORY = r'V:\AIChatBot\StorylineChatbot\uploaded_pdfs'

# Paths to save the index and embeddings
INDEX_PATH = 'faiss_index.index'
EMBEDDINGS_PATH = 'embeddings.json'

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to chunk text
def chunk_text(text, max_tokens=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_tokens = 0

    for sentence in sentences:
        tokens = sentence.split()
        num_tokens = len(tokens)
        if current_tokens + num_tokens <= max_tokens:
            current_chunk += ' ' + sentence
            current_tokens += num_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = num_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to generate embeddings
def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model='text-embedding-ada-002'
        )
        embedding = response['data'][0]['embedding']
        embeddings.append({'text': chunk, 'embedding': embedding})
    return embeddings

# Function to create FAISS index
def create_faiss_index(embeddings_list):
    dimension = len(embeddings_list[0]['embedding'])
    index = faiss.IndexFlatL2(dimension)
    embedding_vectors = np.array([e['embedding'] for e in embeddings_list]).astype('float32')
    index.add(embedding_vectors)
    return index

# Main processing function
def process_pdfs():
    embeddings = []

    for filename in os.listdir(PDF_DIRECTORY):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            pdf_embeddings = generate_embeddings(chunks)
            embeddings.extend(pdf_embeddings)

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Save embeddings and index
    faiss.write_index(index, INDEX_PATH)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    print("Processing complete. Embeddings and index saved.")

if __name__ == '__main__':
    process_pdfs()
