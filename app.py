import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS for your GitHub Pages domain only
CORS(app, origins=["https://akshaysayar.github.io"])

# Configure Gemini AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

def load_cv_context():
    """Load CV data from JSON file"""
    try:
        with open('data/Akshay_Sayar_CV.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"error": "CV data not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in CV data"}

def create_system_prompt(cv_data):
    """Create a comprehensive system prompt with CV context"""
    return f"""You are Akshay Sayar's AI assistant representing him on his CV website. You have access to his complete professional information and should answer questions about his background, experience, skills, and career.

Here is Akshay's complete CV data:
{json.dumps(cv_data, indent=2)}

Instructions:
- Answer questions about Akshay's professional background, experience, skills, education, and projects
- Be conversational, friendly, and professional
- Use emojis appropriately to make responses engaging
- Keep responses concise but informative
- If asked about something not in the CV data, politely mention that information isn't available
- Always respond in plain text format (no markdown formatting)
- Represent Akshay in first person when appropriate (e.g., "I have experience in..." rather than "Akshay has experience in...")
- If someone asks to contact Akshay, provide the contact information from the CV data

Remember: You are representing Akshay Sayar professionally, so maintain a balance between being approachable and maintaining professional credibility.
"""

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with Gemini AI"""
    try:
        # Get the user's message
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Load CV context
        cv_data = load_cv_context()
        if 'error' in cv_data:
            return jsonify({'error': cv_data['error']}), 500
        
        # Create system prompt with CV context
        system_prompt = create_system_prompt(cv_data)
        
        # Combine system prompt with user message
        full_prompt = f"{system_prompt}\n\nUser Question: {user_message}\n\nResponse:"
        
        # Generate response using Gemini
        response = model.generate_content(full_prompt)
        
        # Return plain text response
        return response.text
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return "🚀 Akshay's CV Backend is running!"

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return "👋 Welcome to Akshay Sayar's CV Backend API! Use POST /chat to interact."

if __name__ == '__main__':
    # Check if Google API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable is not set!")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
