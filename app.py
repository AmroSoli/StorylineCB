# app.py

import os
import json
import re
import numpy as np
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tiktoken

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Instantiate the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize conversation history and greeting flag
conversation = []
user_greeted = False  # Flag to track if the user has greeted before

# Path to course embeddings
COURSE_EMBEDDINGS_PATH = 'course_embeddings.json'

# Load course embeddings
with open(COURSE_EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
    course_embeddings = json.load(f)
    # Convert embeddings to numpy arrays
    for e in course_embeddings:
        e['embedding'] = np.array(e['embedding'], dtype='float32')
print("Loaded course embeddings.")

def detect_self_harm(message):
    """
    Placeholder function to detect self-harm related content.
    Implement your self-harm detection logic here.
    """
    # Return True if self-harm content is detected, False otherwise
    return False  # Replace with your implementation

def is_greeting(message):
    """
    Detects if the user's message is a greeting with no other content.
    """
    greetings = ['hello', 'hi', 'hey', 'greetings', 'hola', 'good morning', 'good afternoon', 'good evening']
    pattern_str = r'^\s*(' + '|'.join(map(re.escape, greetings)) + r')[\s!.,]*$'
    greeting_pattern = re.compile(pattern_str, re.IGNORECASE)
    return bool(greeting_pattern.match(message))

def get_relevant_text_chunks(user_input, top_k=3):
    """
    Given the user input, retrieve the most relevant text chunks from the course content.
    """
    # Generate embedding for the user input
    response = client.embeddings.create(
        input=user_input,
        model='text-embedding-ada-002'
    )
    user_embedding = np.array(response.data[0].embedding, dtype='float32')

    # Compute cosine similarity between the user embedding and each course embedding
    similarities = []
    for e in course_embeddings:
        course_embedding = e['embedding']
        similarity = np.dot(user_embedding, course_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(course_embedding))
        similarities.append(similarity)

    # Get the indices of the top_k most similar text chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # Retrieve the corresponding text chunks
    relevant_chunks = [course_embeddings[i]['text'] for i in top_k_indices]
    return relevant_chunks

def truncate_text(text, max_tokens=800):
    """
    Truncate text to a maximum number of tokens.
    """
    encoding = tiktoken.encoding_for_model('gpt-4')
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def num_tokens_from_messages(messages, model="gpt-4"):
    """
    Calculate the total number of tokens in the messages.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        content = message.get('content', '')
        num_tokens += len(encoding.encode(content))
    return num_tokens

def send_to_openai(messages):
    """
    Sends the messages to OpenAI's GPT-4 API and retrieves the assistant's reply.
    """
    MAX_TOTAL_TOKENS = 8192
    MAX_RESPONSE_TOKENS = 500
    MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_RESPONSE_TOKENS

    total_tokens = num_tokens_from_messages(messages)
    if total_tokens > MAX_INPUT_TOKENS:
        # Adjust MAX_RESPONSE_TOKENS to accommodate the input
        MAX_RESPONSE_TOKENS = MAX_TOTAL_TOKENS - total_tokens
        if MAX_RESPONSE_TOKENS < 200:
            raise Exception("Input too long. Please reduce your message or the included context.")

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_RESPONSE_TOKENS,
        )

        # Extract the assistant's reply
        assistant_reply = response.choices[0].message.content.strip()

        return assistant_reply

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        raise e  # Re-raise the exception to be caught in the /chat route

@app.route('/')
def index():
    """
    Route for the main page.
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Route for handling chat messages sent from the user.
    """
    global conversation, user_greeted
    try:
        # Get the user input
        data = request.get_json()
        user_input = data.get('message', '').strip()
        action = data.get('action', '')
        reply_to = data.get('reply_to', '').strip()

        # Handle actions: simplify, quiz, reply
        if action:
            # Handle simplify and quiz actions
            if action == 'simplify':
                if not reply_to:
                    return jsonify({'status': 'error', 'reply': 'No content to simplify.'})
                messages = [
                    {"role": "system", "content": (
                        "You are a helpful assistant that simplifies complex text for better understanding. "
                        "Provide clear and concise explanations using only the course content provided."
                    )},
                    {"role": "system", "content": f"Course Content:\n\n{reply_to}"},
                    {"role": "user", "content": "Please simplify the above content."}
                ]
                assistant_reply = send_to_openai(messages)
                return jsonify({'status': 'success', 'reply': assistant_reply})

            elif action == 'quiz':
                if not reply_to:
                    return jsonify({'status': 'error', 'reply': 'No content to generate a quiz from.'})
                # Generate a quiz based on the reply_to content
                messages = [
                    {"role": "system", "content": (
                        "You are a helpful assistant that generates quizzes from provided content. "
                        "When asked to generate a quiz, output only the JSON data without any additional text."
                    )},
                    {"role": "system", "content": f"Course Content:\n\n{reply_to}"},
                    {"role": "user", "content": (
                        "Create a short quiz with 3 multiple-choice questions based on the above content. "
                        "Provide the quiz in JSON format as a list of questions, where each question is a dictionary "
                        "with keys 'question', 'options', and 'answer'. Do not include any text outside of the JSON data."
                    )}
                ]
                assistant_reply = send_to_openai(messages)

                # Try to parse the assistant's reply as JSON
                try:
                    quiz_data = json.loads(assistant_reply)
                    # Return the quiz data to the frontend
                    return jsonify({'status': 'success', 'quiz': quiz_data})
                except json.JSONDecodeError:
                    # If parsing fails, return an error message
                    return jsonify({'status': 'error', 'reply': 'Failed to generate quiz. Please try again.'})

            elif action == 'reply':
                # Handle reply action if needed
                pass  # Currently, no special handling required

        else:
            # Regular message processing
            # Check for self-harm content
            if detect_self_harm(user_input):
                assistant_reply = (
                    "I'm sorry to hear that you're feeling this way. "
                    "Please consider reaching out to a mental health professional or a trusted person in your life."
                )
                return jsonify({'reply': assistant_reply})

            # Handle greetings
            if is_greeting(user_input):
                if not user_greeted:
                    assistant_reply = "Hello, how may I help? 😊"
                    user_greeted = True
                else:
                    assistant_reply = "Hello, I'm still here. Do you have any questions? 😊"
                conversation.append({"role": "user", "content": user_input})
                conversation.append({"role": "assistant", "content": assistant_reply})
                return jsonify({'reply': assistant_reply})

            # Update conversation
            conversation.append({"role": "user", "content": user_input})

            # Retrieve relevant text chunks
            relevant_chunks = get_relevant_text_chunks(user_input, top_k=3)
            # Truncate chunks
            truncated_chunks = [truncate_text(chunk, max_tokens=800) for chunk in relevant_chunks]
            # Combine the relevant chunks into a single string
            context_text = "\n\n".join(truncated_chunks)

            # Construct messages
            messages = [
                {"role": "system", "content": (
                    "You are a helpful assistant called CivilTutor. "
                    "Use the following course content to answer the user's questions. "
                    "If the answer is not in the course content, politely inform the user that you can only provide information from the course. "
                    "Do not provide any information beyond the course content."
                )},
                {"role": "system", "content": f"Course Content:\n\n{context_text}"},
                {"role": "user", "content": user_input}
            ]

            # Send to OpenAI API
            assistant_reply = send_to_openai(messages)

            # Update conversation
            conversation.append({"role": "assistant", "content": assistant_reply})

            # Return reply
            return jsonify({'reply': assistant_reply})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return jsonify({'reply': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
