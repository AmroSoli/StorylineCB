# app.py

import os
import json
import re
import numpy as np
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tiktoken
import traceback

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

# Load course embeddings at startup
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

    # Efficient similarity calculation
    embeddings_matrix = np.array([e['embedding'] for e in course_embeddings])
    similarities = np.dot(embeddings_matrix, user_embedding) / (
        np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(user_embedding)
    )

    # Get the indices of the top_k most similar text chunks
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve the corresponding text chunks
    relevant_chunks = [course_embeddings[i]['text'] for i in top_k_indices]
    return relevant_chunks

def truncate_text(text, max_tokens=800):
    """
    Truncate text to a maximum number of tokens.
    """
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """
    Calculate the total number of tokens in the messages.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        content = message.get('content', '')
        num_tokens += len(encoding.encode(content))
    return num_tokens

def send_to_openai(messages, max_response_tokens=1000):
    """
    Sends the messages to OpenAI's API and retrieves the assistant's reply.
    """
    MAX_TOTAL_TOKENS = 4096  # For gpt-3.5-turbo
    MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - max_response_tokens

    total_tokens = num_tokens_from_messages(messages)
    if total_tokens > MAX_INPUT_TOKENS:
        # Adjust max_response_tokens to accommodate the input
        max_response_tokens = MAX_TOTAL_TOKENS - total_tokens
        if max_response_tokens < 200:
            raise Exception("Input too long. Please reduce your message or the included context.")

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=max_response_tokens,
        )

        # Extract the assistant's reply
        assistant_reply = response.choices[0].message.content.strip()

        return assistant_reply

    except Exception as e:
        error_message = f"An error occurred in send_to_openai: {str(e)}"
        traceback_str = traceback.format_exc()
        print(error_message)
        print(traceback_str)
        raise e  # Re-raise the exception to be caught in the /chat route

def extract_json(text):
    """
    Extracts JSON content from text.
    """
    json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
    if json_match:
        return json_match.group()
    else:
        return None

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
                        "When asked to generate a quiz, you should output only the JSON data and nothing else. "
                        "Do not include any explanations, apologies, or additional text."
                    )},
                    {"role": "system", "content": f"Course Content:\n\n{reply_to}"},
                    {"role": "user", "content": (
                        "Create a short quiz with 5 multiple-choice questions based on the above content. "
                        "Provide the quiz in strict JSON format as a list of questions. "
                        "Each question should be a dictionary with the keys 'question', 'options', and 'answer'. "
                        "Ensure that the JSON is valid and parsable. Do not include any text outside of the JSON data."
                    )}
                ]

                try:
                    assistant_reply = send_to_openai(messages, max_response_tokens=1000)
                    json_text = extract_json(assistant_reply)
                    if not json_text:
                        raise ValueError("No JSON content found in assistant's reply.")
                    quiz_data = json.loads(json_text)
                    return jsonify({'status': 'success', 'quiz': quiz_data})
                except (json.JSONDecodeError, ValueError) as e:
                    error_message = f"JSON parsing error: {str(e)}"
                    print("Assistant's reply that caused parsing error:")
                    print(assistant_reply)
                    print(error_message)
                    return jsonify({'status': 'error', 'reply': 'Failed to generate quiz. Please try again.'})
                except Exception as e:
                    error_message = f"Unexpected error: {str(e)}"
                    print(error_message)
                    return jsonify({'status': 'error', 'reply': 'An error occurred while processing your request.'})

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
        error_message = f"An error occurred while processing your request: {str(e)}"
        traceback_str = traceback.format_exc()
        print(error_message)
        print(traceback_str)
        return jsonify({'reply': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)