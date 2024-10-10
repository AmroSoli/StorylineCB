# app.py
# This is the main Python file for your chatbot application.

import os
import json
import re
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize conversation history
conversation = []

# Paths to the folders containing PDFs
PDF_FOLDER_PATH = 'pdfs'            # Default PDFs folder
UPLOADED_PDFS_PATH = 'uploaded_pdfs'  # Uploaded PDFs folder

# Ensure the 'uploaded_pdfs' directory exists
if not os.path.exists(UPLOADED_PDFS_PATH):
    os.makedirs(UPLOADED_PDFS_PATH)

# Initialize course content
course_content = ""

# Define the chapters/modules
chapters = [
    "1. Introduction to Construction Workplace Health and Safety",
    "2. Planning and Preparation",
    "3. Hazard Identification and Control",
    "4. Worksite Safety and Health",
    "5. Safe Work Methods and Procedures"
]

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdfs(folder_path):
    """
    Extracts text from all PDF files in the specified folder.
    """
    pdf_text = ""

    # Check if the folder exists
    if not os.path.exists(folder_path):
        return pdf_text

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                # Extract text from each page
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + "\n"

    return pdf_text

def update_course_content():
    """
    Updates the global course content variable by extracting text from PDFs.
    """
    global course_content
    # Extract text from PDFs in both 'pdfs' and 'uploaded_pdfs' folders
    content_pdfs = extract_text_from_pdfs(PDF_FOLDER_PATH)
    uploaded_pdfs = extract_text_from_pdfs(UPLOADED_PDFS_PATH)
    course_content = content_pdfs + "\n" + uploaded_pdfs

# Initialize course content
update_course_content()

def detect_self_harm(message):
    """
    Detects if the message contains self-harm related content.
    """
    # Define keywords related to self-harm
    self_harm_keywords = [
        r'\bkill myself\b',
        r'\bsuicide\b',
        r'\bself[-\s]?harm\b',
        r'\bend my life\b',
        r'\bhurt myself\b',
        # Add more keywords as needed
    ]
    pattern = re.compile('|'.join(self_harm_keywords), re.IGNORECASE)
    return bool(pattern.search(message))

def is_greeting(message):
    """
    Detects if the user's message is a greeting.
    """
    greetings = ['hello', 'hi', 'hey', 'greetings', 'hola', 'good morning', 'good afternoon', 'good evening']
    # Create a regex pattern to match greetings as whole words, case-insensitive
    pattern = r'\b(' + '|'.join(map(re.escape, greetings)) + r')\b'
    return re.search(pattern, message, re.IGNORECASE) is not None

def construct_input_prompt(conversation_history):
    """
    Constructs the input prompt for the Llama model based on the conversation history.
    """
    # System prompt including the course content and strict instructions
    system_prompt = (
        "You are a helpful assistant called CivilTutor for an e-learning course about construction safety. "
        "Your role is to assist learners by answering questions strictly based on the course content provided below. "
        "Do not use any external information or your own knowledge. "
        "If the answer is not in the course content, politely inform the user that the information is not available in the course material. "
        "When the user greets you during the conversation, respond appropriately, but do not repeat the initial greeting message. "
        "After providing an explanation, include the following message in bold at the end: "
        "'Would you like me to simplify this further? If so, type \"Simplify\" :)' "
        "If the user types 'Simplify', then provide a simpler version of your last explanation. "
        "Continue simplifying upon each 'Simplify' command until it can no longer be simplified. "
        "Please provide clear, concise answers and format your responses using markdown for better readability. "
        "Use headings, bold text, and bullet points where appropriate. "
        "Do not include any content unrelated to the course content.\n\n"
        "### Chapters:\n"
        f"{chr(10).join(chapters)}\n\n"
        "### Course Content:\n"
        f"{course_content}\n\n"
    )

    # Build the conversation history into a single string
    conversation_text = system_prompt
    for msg in conversation_history:
        role = msg['role'].capitalize()  # 'User' or 'Assistant'
        content = msg['content']
        conversation_text += f"{role}: {content}\n"

    # Add the assistant's prompt to indicate that the assistant should respond
    conversation_text += "Assistant:"

    return conversation_text

def send_to_ollama(input_text):
    """
    Sends the input prompt to the Llama model API and retrieves the assistant's reply.
    """
    try:
        # Make a POST request to the Ollama API with the input prompt
        response = requests.post(
            'http://localhost:11434/api/generate',  # API endpoint
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            json={
                "model": "llama3.2:latest",
                "prompt": input_text,
                "temperature": 0.7,
                "max_tokens": 500,
            },
            stream=True
        )
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Initialize the assistant's reply
        assistant_reply = ""

        # Iterate over the streamed response lines
        for line in response.iter_lines():
            if line:
                # Decode the line from bytes to string
                line_str = line.decode('utf-8')
                try:
                    # Parse the JSON data from each line
                    data = json.loads(line_str)
                    # Append the 'response' field to the assistant's reply
                    assistant_reply += data.get('response', '')
                except json.JSONDecodeError:
                    # Skip lines that are not valid JSON
                    continue

        # Return the assistant's reply, stripped of leading/trailing whitespace
        return assistant_reply.strip()

    except requests.exceptions.RequestException as e:
        # Handle request exceptions
        print(f"An error occurred while communicating with the API: {str(e)}")
        return "I'm sorry, but I'm unable to respond at the moment."

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
    global conversation
    try:
        # Get the user input
        data = request.get_json()
        user_input = data.get('message', '').strip()

        # Check for self-harm content
        if detect_self_harm(user_input):
            assistant_reply = (
                "I'm sorry to hear that you're feeling this way. "
                "Please consider reaching out to a mental health professional or a trusted person in your life. "
                "In Australia, you can contact Lifeline at 13 11 14 for support."
            )
            return jsonify({'reply': assistant_reply})

        # Check if the user wants to simplify the last explanation
        if user_input.lower() == 'simplify':
            # Get the last assistant's message
            last_assistant_message = ''
            for msg in reversed(conversation):
                if msg['role'] == 'assistant':
                    last_assistant_message = msg['content']
                    break
            if last_assistant_message:
                # Prepare prompt to simplify
                simplify_prompt = (
                    "Simplify the following explanation further into simpler terms. "
                    "Continue simplifying until it can no longer be simplified. "
                    "Only use the information provided.\n\n"
                    f"Explanation:\n{last_assistant_message}"
                )
                assistant_reply = send_to_ollama(simplify_prompt)
                # Append the bolded message
                assistant_reply += "\n\n**Would you like me to simplify this further? If so, type 'Simplify' :)**"
                # Update conversation
                conversation.append({"role": "user", "content": user_input})
                conversation.append({"role": "assistant", "content": assistant_reply})
                return jsonify({'reply': assistant_reply})
            else:
                assistant_reply = "I'm sorry, but I don't have anything to simplify."
                return jsonify({'reply': assistant_reply})

        # Handle greetings during the conversation
        if is_greeting(user_input):
            assistant_reply = "Hello, I'm still here. Do you have any questions?"
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": assistant_reply})
            return jsonify({'reply': assistant_reply})

        # Add user's message
        conversation.append({"role": "user", "content": user_input})

        # Construct prompt and get assistant's reply
        input_prompt = construct_input_prompt(conversation)
        assistant_reply = send_to_ollama(input_prompt)

        # Append the bolded message if not already included
        if "**Would you like me to simplify this further?" not in assistant_reply:
            assistant_reply += "\n\n**Would you like me to simplify this further? If so, type 'Simplify' :)**"

        # Update conversation
        conversation.append({"role": "assistant", "content": assistant_reply})

        # Return reply along with a success flag
        return jsonify({'reply': assistant_reply, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handles PDF file uploads from the user.
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file part in the request.'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No selected file.'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = UPLOADED_PDFS_PATH
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Update course content
        update_course_content()

        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'error': 'Only PDF files are allowed.'})

if __name__ == '__main__':
    app.run(debug=True)
