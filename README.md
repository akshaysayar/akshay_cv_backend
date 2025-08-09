# Akshay CV Backend

A Flask backend API for Akshay Sayar's CV website that integrates with Google's Gemini LLM to provide intelligent responses about professional background and experience.

## Features

- 🤖 AI-powered chat endpoint using Google Gemini LLM
- 📄 Context-aware responses based on CV data
- 🔒 CORS protection (configured for GitHub Pages)
- 🚀 Ready for Render deployment
- 💬 Plain text responses with emoji support

## API Endpoints

### POST /chat
Send a message to get AI-powered responses about Akshay's professional background.

**Request:**
```json
{
  "message": "Tell me about your experience"
}
```

**Response:**
```
Plain text response with emojis about the requested information
```

### GET /health
Health check endpoint.

### GET /
Welcome message.

## Setup

### Option 1: Using Poetry (Recommended)

1. Clone the repository
2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_actual_google_api_key_here
     ```

5. Update the CV data in `data/Akshay_Sayar_CV.json` with your actual information

6. Run locally:
   ```bash
   poetry run python app.py
   ```

### Option 2: Using pip

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_actual_google_api_key_here
     ```

4. Update the CV data in `data/Akshay_Sayar_CV.json` with your actual information

5. Run locally:
   ```bash
   python app.py
   ```

## Deployment on Render

### With Poetry (Recommended)
1. Connect your GitHub repository to Render
2. Render will automatically detect `pyproject.toml` and use Poetry
3. Set the environment variable `GOOGLE_API_KEY` in Render dashboard
4. Render will automatically install dependencies and deploy

### With pip (Alternative)
1. Connect your GitHub repository to Render
2. Set the environment variable `GOOGLE_API_KEY` in Render dashboard
3. Render will automatically detect the Python app and deploy using `requirements.txt`

**Note**: Poetry provides better dependency resolution and more reliable deployments, so it's the recommended approach.

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `PORT`: Port for the application (automatically set by Render)

## CORS Configuration

The API is configured to only accept requests from:
- `https://akshaysayar.github.io`

## CV Data Structure

Update `data/Akshay_Sayar_CV.json` with your actual CV information. The JSON should include:
- Personal information
- Work experience
- Education
- Skills
- Projects
- Certifications
- Languages
- Interests
