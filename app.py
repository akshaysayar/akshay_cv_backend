import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS for your GitHub Pages domain and local development
CORS(app, origins=["https://akshaysayar.github.io", "http://localhost:8000"])

# Configure Gemini AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

akshay_cv_context = {
  "Professional_work": [
    {
      "title": "Software Engineer",
      "start": "Nov 2022",
      "end": "July 2025",
      "institution": "Propylon",
      "location": "Dublin, Ireland",
      "type": "work",
      "content": [
        "Developed a conversational Retrieval-Augmented Generation (RAG) system using LangGraph and Claude Sonnet LLM, providing accurate, context-aware answers to complex legal research queries.",
        "Designed and implemented a custom indexing and retrieval algorithm integrated with OpenSearch, significantly improving semantic search precision and retrieval performance.",
        "Engineered dynamic query understanding, enabling the system to extract entities and filters from user input to narrow document search space and improve retrieval precision.",
        "Built intelligent query rewriting and source document reranking to maximize response accuracy and relevance.",
        "Applied Guardrails to enforce output safety, prevent hallucinations, and maintain strict response formats.",
        "Developed ML model to classify laws using Zero-shot classification.",
        "Designed and built a comprehensive system to generate an interactive timeline of laws and regulations by processing structured legal data from XML files, leveraging NLP, machine learning, and custom frameworks.",
        "Developed custom NLP pipelines to extract and classify legal entities, events, and relationships from unstructured text.",
        "Automated the entire pipeline using GitHub Actions for CI/CD, enabling smooth updates and reproducibility.",
        "Deployed the system and models to AWS Cloud for scalable, production ready performance.",
        "Mentored Trinity College students on AI proof-of-concept projects for internship (SWEng Program) for three years.",
        "Volunteered to teach math and programming to third-grade students at The Good Shepherd National School (Time To Count Program).",
        "I have taken initiative to solve problems creatively and efficiently, and I have a strong passion for learning and exploring new technologies.",
        "I proactively look for opportunities to improve processes and systems, and I am always looking for ways to contribute to the success of the team and the organization."
      ]
    },
    {
      "title": "Machine Learning Engineer",
      "start": "Feb 2022",
      "end": "June 2022",
      "institution": "Ultrasound Ireland",
      "location": "Dublin, Ireland",
      "type": "work",
      "content": [
        "Identified problem and solution to automate data entry of Hospital form Information into SQL.",
        "Lead the team of 5 from different backgrounds to build a product.",
        "Created a pipeline to extract data from the form and store it in SQL.",
        "OCR to automate data entry of Hospital form Information into SQL.",
        "Trained YOLO V5 model to recognize special marking techniques on the form.",
        "Custom Trained Azure Form Recognizer for frequent form for high accuracy.",
        "Ensemble AWS Textract, Azure custom trained Form Recognizer model and transferred learned YOLO V5.",
        "Used open-source Tesseract to classify forms to select from Azure and AWS.",
        "Created API for the product."
      ]
    },
    {
      "title": "Master of Science in Data Analytics",
      "institution": "National College of Ireland",
      "location": "Dublin, Ireland",
      "start": "Sep 2021",
      "end": "Oct 2022",
      "type": "study",
      "content": [
        "Degree obtained October 2022",
        "Graduated with First Class Honors (1:1).",
        "Build ML Framework for predicting Empathy using eye tracking and Pupil Dilation for thesis. Experimented and gathered the data myself.",
        "Project: ML framework to detect high empathy using eye-tracking and pupil dilation.",
        "Problem: Questionnaire responses can be manipulated; objective physiological signals offer complementary evidence of empathic response.",
        "Data collection: 53 participants watched six validated sad-story videos while we recorded point-of-gaze and pupil radius/diameter.",
        "Preprocessing & tools: Used deep-learning/vision tools (ellseg, YOLOv5, dlib) to detect facial regions, eyes, and pupil contours; synchronized gaze and pupil time series with stimulus timestamps.",
        "Feature engineering: Extracted gaze heatmaps, fixation statistics, time-based pupil dilation percentiles, temporal trends, and aggregated summary metrics.",
        "Labels: Target empathy scores derived from a validated empathy questionnaire administered alongside the experiment.",
        "Models: Trained and compared Random Forest, XGBoost, Gradient Boosting, and Logistic Regression on full and reduced feature sets.",
        "Results: Logistic Regression performed best with 89% accuracy; analyses show pupil dilation is the strongest predictor followed by gaze features.",
        "Impact & application: Framework provides a scalable, objective signal to augment screening for empathy-critical roles (e.g., nursing, counseling) and supports fairer, data-driven recruitment.",
        "Reproducibility & governance: Pipeline includes reproducible preprocessing, clear feature documentation, and evaluation metrics for model validation and ethical deployment.",
        "Included in the Dean’s Honor’s List for academic excellence.",
        "ENFUSE Competition - 3rd Position - (Inter-College Level) and Winner (College Level)."
      ]
    },
    {
      "title": "Data Scientist | GS Lab| Pune",
      "start": "Dec 2019",
      "end": "Aug 2021",
      "institution": "GS Lab",
      "location": "Pune, India",
      "type": "work",
      "content": [
        "Worked closely with IBM Watson AIOps research team.",
        "Trained and fine-tuned AIOps model on various client data on IBM cloud.",
        "Lead the team in building Network Intrusion Detection System using ML. - Model was trained on eight different cyber-attacks data.",
        "Trained on various ML models like RandomForest, XGBoost, SVM, Logistic regression etc. with thorough hyperparameter tuning.",
        "Build an application for detecting real-time network intrusion detection.",
        "Build an Anomaly detection system for IoT devices.",
        "Analyzed the data and its statistical properties.",
        "Applied various models like holt winter, SARIMA, GRU, and LSTM.",
        "Created stacked LSTM model which showed the highest accuracy.",
        "Led a team of four to build a multi-threaded RFID-to-IBM Cloud system enabling remote device control and real-time storage of EPC, TID, and user memory data."
      ]
    },
    {
      "title": "Data Scientist | Cognizant | Pune",
      "start": "Feb 2021",
      "end": "Nov 2021",
      "institution": "Cognizant",
      "location": "Pune, India",
      "type": "work",
      "content": [
        "Worked for a Banking client.",
        "Created interactive data visualizations and dashboards in Power BI to support data-driven insights and decision-making.",
        "Created backend API for data extraction from Hive Database and upstream applications.",
        "A/B testing of models."
      ]
    },
    {
      "title": "QA Automation Engineer | Cognizant | Pune",
      "start": "Nov 2019",
      "end": "Feb 2021",
      "institution": "Cognizant",
      "location": "Pune, India",
      "type": "work",
      "content": [
        "Handled multiple projects for testing for a Banking client.",
        "Created end-to-end generic Regression Script for testing API and data using Python and input data via MS Excel which tests data on the various levels depending on parameters provided in MS Excel. ",
        "Automated all testing scenarios at the end of each release using Python and ROBOT Framework.",
        "Responsible for QA signoff for multiple applications within the project.",
        "Represented the RBC project in the Innovation competition (had built Automated test-case documentation using Python)."
      ]
    },
    {
      "title": "Bachelor of Engineering, Mechanical Engineering",
      "institution": "University of Pune",
      "location": "Pune, India",
      "start": "July 2012",
      "end": "June 2016",
      "type": "study",
      "content": [
        "Degree obtained July 2016",
        "Graduated with First Class with Distinction.",
        "Designed and Build Compressed Air Engine."
      ]
    }
  ],
  "Personal Projects": {
    "content": [
      "ML Framework for predicting Empathy using eye tracking and Pupil Dilation (Aug 2022)",
      "Developed a machine learning framework to predict empathy using eye-tracking and pupil dilation data, integrating deep learning tools (ELLSEG, YOLOv5, dlib) and psychological analysis. Achieved 89% accuracy using Logistic Regression, demonstrating strong potential for applications in empathetic candidate screening.",
      "Free Canvas (Feb 2022)",
      "\tFree Canvas (Augmented Reality) is a drawing tool for children of age 2-8 where they can interact with webcam and draw figures by joining dots and improving creativity. \nFree canvas works by recognizing hand and finger points and children can pinch a dot and create a line.",
      "Agentic Chatbot (March 2021)",
      "\tDesigned and created a chatbot for personal use which can perform certain tasks on Ubuntu OS. \nThe data was made from scratch, and the model was trained on the data.\nA speech to text module from google was also used on top of the above model and it acted as a personal assistant. (Siri, Alexa etc.)",
      "Snake Game for Random Forest (Jan 2021)",
      "\tDesigned and built a snake game on Webcam's screen where you can see yourself (like a mirror) and the snake playing the game itself.\nMultiple models were trained to play the game, Random Forest gave the best accuracy, and we can see the Random Forest model playing the game. (It's addictive watching it). \nData for training the model was generated from scratch by playing 100 games manually."
    ]
  },
  "Skills": {
    "content": [
      "Python (Programming Language), Natural Language Processing (NLP), Artificial Intelligence (AI), Generative Artificial Intelligence, LangGraph (Gen AI), Image Processing, Prompt Engineering, Transformers, Django, Huggingface, Redis, FAISS, Postgres, Memory Store, OpenSearch, Github, Localstack, LXML, Terraform, Docker, Make, AWS, AWS Lambda, Machine Learning, Retrieval-Augmented Generation (RAG), Semantic Search, YOLO, Confluence, Jira, Hive, Pyspark, scikit, sklearn, XGBoost, Deep learning, Data Science, Computer Vision, Pandas, SQL, MS Excel, Java, Seaborn, Jenkins, Ubuntu, Linux, WSL, JSON, XML, spacy, beautifulSoup, matplotlib, HTML, numpy, linear regression, logistic regression, PCA, CNN, Random Forest, Request, Pattern recognition, regex, data cleaning, data extraction, neural network, ROC, MYSQL, OOP, github workflows, github actions, unit test, scenario tests, pytest, tmux."
    ]
  },
  "About": {
    "content": [
      "I play with code and AI to build smart, meaningful solutions. I'm a technophile at heart and a passionate learner with 7.5+ years of experience as a Data Scientist, AI Engineer, Software Engineer, Data Engineer. I aspire to be a philanthropist — using technology to improve lives, especially for children, and to drive positive, lasting impact through innovation. ",
      "I like solving problems, I like coding, I love maths, I love technology, reading books, listening music, playing games.",
      "What resume cant teach you about me is that I am a very curious person, I like to learn new things, I like to explore new technologies, I like to solve problems and I like to build things. I am a very open-minded person and I like to work with people from different backgrounds and cultures. I believe that diversity is the key to innovation and creativity.",
      "I am happy young person, I like to spread happiness, help those in needs, and I like to make a difference in the world. I am a very positive person and I believe that anything is possible if you put your mind to it. I am a very hard-working person and I always strive to do my best in everything I do.",
      "I would say that my biggest strength is disclipline, I am very disciplined person and I always try to do things in a systematic way. I am also a very organized person and I like to keep things in order. I believe that discipline and organization are the keys to success.",
      " I am very polite, respectful and I always try to be helpful to others. I believe that kindness is the key to happiness and I always try to spread happiness wherever I go."
    ]
  },
  "between_the_lines": {
    "content": [
      "Strategic Technologist: Thinks beyond individual components to design end-to-end systems (embedding pipelines, retrieval layers, orchestration, deployment) that solve real business problems.",
      "End-to-End Ownership: Comfortable owning the full ML lifecycle — data ingestion, preprocessing, model training, evaluation, CI/CD deployment, monitoring, versioning, and governance.",
      "Multi-Domain Versatility: Picks up domain knowledge quickly and applies ML across regulated and complex areas like telecom, legal NLP, life sciences, and empathy-detection research.",
      "Pragmatic Cloud Engineer: Chooses cloud and infra technologies for business fit; hands-on with AWS services (SageMaker, Fargate, OpenSearch, RDS, SQS, Bedrock) and GCP patterns when appropriate.",
      "Production-Grade Focus: Prioritises reliability, observability, and reproducibility — builds containerised microservices, instrumented monitoring (Prometheus/Grafana/CloudWatch), and robust MLOps pipelines.",
      "Impact-Oriented Communicator: Frames technical work in terms of product and business outcomes (faster audits, better recruitment, automated telecom rApps), enabling cross-team alignment with non-technical stakeholders.",
      "Curiosity & Depth: Digs into internals (LangChain/LangGraph orchestration, embedding strategies, Django internals, tool security) to deliver thoughtful, well-reasoned solutions rather than superficial fixes.",
      "Resilient, Iterative Learner: Uses interview and project feedback as data — iterates rapidly, refines approaches, and demonstrates a growth mindset when improving designs and implementations.",
      "Comfortable with Complexity: Prefers multi-service, cloud-native architectures with strict compliance/observability needs and thrives in technically complex, high-stakes environments."
    ]
  },
  "Contact": {
    "Email": "akshaysayar@gmail.com",
    "Phone": "+353899666388",
    "LinkedIn": "https://www.linkedin.com/in/akshay-sayar/",
    "GitHub": "https://github.com/akshaysayar",
    "Address": "63, Block 2, O'Neill Court, Dublin 13"
  }
}

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
- Default to 3–5 short bullet points (use '-' bullets). If the user explicitly asks for a paragraph, use 3–5 short sentences.
- Be conversational, friendly, and professional.
- Use simple smileys/emojis sparingly to make responses engaging (e.g., 🙂😊👍); at most one per bullet.
- Keep responses concise; avoid long explanations.
- If asked about something not in the CV data, politely say it's not available.
- Refer to Akshay in third person (e.g., "Akshay has experience in ...", not "I have experience ...").
- Do not share personal contact info unless explicitly requested; if requested, use the Contact section from the CV only.
- Do not invent projects, employers, or metrics not present in the CV.
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
        
        # Generate response using Gemini with concise generation settings
        response = model.generate_content(
            full_prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 300
            }
        )
        
        # Return plain text response with safe fallback
        text = getattr(response, 'text', '') or ''
        text = text.strip()
        if not text:
            return "Sorry, I couldn't generate a response right now. Please try again. 🙂"
        return text
        
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
