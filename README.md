# Context-Specific Machine Translation to Arabic

A modern, full-stack web application for context-aware machine translation to Arabic, supporting English and French as source languages. The system features a React-based frontend with a Flask API backend, designed to provide high-quality, domain-specific translations.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [Demo Video](#demo-video)

## Project Overview

This project implements a machine translation system specifically designed for translating to Arabic. The application provides an intuitive web interface where users can input text in English or French and receive context-aware translations in Arabic. The system supports domain-specific translations and includes features like file upload, translation history, and language detection.

## Repository Structure

```
Context_Specific_Machine_Translation_to_Arabic_Language/
│
├── app/                                # Main application directory
│   ├── api/                           # Flask backend API
│   │   ├── __init__.py
│   │   ├── app.py                     # Flask application factory
│   │   ├── config.py                  # Configuration settings
│   │   ├── routes.py                  # API route definitions
│   │   └── rag_service.py             # Translation service integration
│   │
│   ├── src/                           # React frontend source
│   │   ├── components/                # Reusable UI components
│   │   ├── contexts/                  # React context providers
│   │   ├── hooks/                     # Custom React hooks
│   │   ├── pages/                     # Page components
│   │   ├── types/                     # TypeScript type definitions
│   │   ├── App.tsx                    # Main application component
│   │   └── main.tsx                   # Application entry point
│   │
│   ├── public/                        # Static assets
│   ├── .env                           # Environment variables (not in git)
│   ├── .env.example                   # Environment variables template
│   ├── package.json                   # Node.js dependencies
│   ├── requirements.txt               # Python dependencies
│   ├── vite.config.ts                 # Vite configuration
│   ├── tailwind.config.ts             # Tailwind CSS configuration
│   └── index.html                     # HTML entry point
│
├── Evaluation/                        # Translation evaluation system
│   ├── Data/                          # Test datasets
│   │   ├── english.csv                # English-Arabic test pairs
│   │   └── french.csv                 # French-Arabic test pairs
│   ├── Results/                       # Evaluation results (auto-generated)
│   ├── evaluate_translation.py        # Main evaluation script
│   ├── evaluate_enhanced.py           # Enhanced evaluation with detailed metrics
│   ├── requirements.txt               # Evaluation dependencies
│   └── README.md                      # Evaluation documentation
│
├── RAG/                               # RAG system (NOT YET IMPLEMENTED)
│   ├── glossary-system/               # Glossary lookup service (planned)
│   ├── prompt-construction/           # Prompt engineering service (planned)
│   ├── RAG-SYSTEM/                    # Retrieval-augmented generation (planned)
│   ├── INTEGRATION/                   # Integration guides (planned)
│   └── README.md                      # RAG system documentation
│
├── RLHF/                              # RLHF system (NOT YET IMPLEMENTED)
│   ├── 0_config_setup.ipynb           # Configuration notebook (planned)
│   ├── 1_synthetic_data_generation.ipynb  # Data generation (planned)
│   ├── 2_reward_model_training.ipynb  # Reward model (planned)
│   ├── 3_ppo_optimization.ipynb       # PPO training (planned)
│   ├── 4_inference_user_interaction.ipynb # Inference (planned)
│   └── 5_human_preference_finetuning.ipynb # Fine-tuning (planned)
│
├── start_integrated_system.sh         # Script to start all services
├── stop_integrated_system.sh          # Script to stop all services
└── README.md                          # This file
```

### Directory Explanations

**app/**: Contains the complete web application with both frontend and backend code. The React frontend provides the user interface while the Flask API handles translation requests and service integration.

**Evaluation/**: Includes scripts and test datasets for evaluating translation quality using standard metrics (BLEU, CHRF+). Contains both English-Arabic and French-Arabic test pairs.

**RAG/**: Directory structure for a planned Retrieval-Augmented Generation system to enhance translation quality with domain-specific context. **This feature is not yet implemented.**

**RLHF/**: Directory structure for a planned Reinforcement Learning from Human Feedback system to fine-tune translation models based on user preferences. **This feature is not yet implemented.**

## Features

- **Real-time Translation**: Translate text from English or French to Arabic
- **Automatic Language Detection**: Automatically identifies source language
- **Multi-Service Support**: Integrated with DeepL (free tier) and OpenRouter APIs
- **File Upload**: Upload and translate content from PDF and DOCX files
- **Translation History**: View and manage previous translations
- **Domain-Specific Translation**: Support for various domains (technology, healthcare, legal, etc.)
- **Responsive Design**: Mobile-first UI design with modern components
- **Error Handling**: Robust error handling with graceful fallbacks

## Technology Stack

### Frontend

- **React 18** with TypeScript for type-safe component development
- **Vite** for fast development and optimized production builds
- **shadcn/ui** for pre-built, accessible UI components
- **Tailwind CSS** for utility-first styling
- **React Router** for client-side routing
- **TanStack Query** for efficient data fetching and caching

### Backend

- **Flask 2.3.3** as the web framework
- **Flask-CORS** for handling cross-origin requests
- **Python 3.10+** for backend logic
- **python-dotenv** for environment variable management
- **PyMuPDF** for PDF text extraction
- **python-docx** for DOCX file processing

### Translation Services

- **DeepL API** (Free tier, default) - 500,000 characters/month
- **OpenRouter API** (Optional, paid) - For advanced model access

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.10 or higher) - [Download](https://www.python.org/)
- **npm** or **yarn** - Comes with Node.js
- **pip** - Python package installer

Optional:
- **Git** - For cloning the repository
- **A code editor** - VS Code, Sublime Text, etc.

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbdelrahimAnesCHABIRA/NLP-PROJECT.git
cd Context_Specific_Machine_Translation_to_Arabic_Language
```

### Step 2: Backend Setup

Navigate to the app directory and set up the Python environment:

```bash
cd app
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_PORT=5002
CORS_ORIGINS=http://localhost:8080,http://localhost:5173

# Translation Service Configuration
TRANSLATION_SERVICE=deepl

# DeepL API Configuration (FREE - DEFAULT)
# Get your free API key at: https://www.deepl.com/pro-api
# Free tier: 500,000 characters/month (no credit card required)
DEEPL_API_KEY=your_deepl_api_key_here

# OpenRouter API Configuration (PAID - OPTIONAL)
# Get your API key at: https://openrouter.ai/
# Only needed if you set TRANSLATION_SERVICE=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4-turbo
OPENROUTER_FALLBACK_MODEL=meta-llama/llama-3.2-3b-instruct:free
```

**Important**: Replace `your_deepl_api_key_here` with your actual DeepL API key. You can get a free API key at [DeepL Pro API](https://www.deepl.com/pro-api).

### Step 3: Frontend Setup

While still in the app directory, install Node.js dependencies:

```bash
npm install
```

This will install all required frontend packages including React, Vite, Tailwind CSS, and UI components.

### Step 4: Verify Installation

Check that all dependencies are installed correctly:

```bash
# Check Python packages
pip list | grep Flask

# Check Node packages
npm list --depth=0
```

## Running the Application

There are multiple ways to run the application depending on your needs:

### Option 1: Run Full Stack (Recommended)

Run both frontend and backend together using the integrated script:

```bash
cd app
npm run dev:full
```

This will start:
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:5002

### Option 2: Run Services Separately

#### Terminal 1 - Start Backend:

```bash
cd app
npm run api
```

The API will be available at http://localhost:5002

#### Terminal 2 - Start Frontend:

```bash
cd app
npm run dev
```

The frontend will be available at http://localhost:8080

### Option 3: Use Shell Scripts (Advanced)

From the project root directory:

```bash
# Start the integrated system
./start_integrated_system.sh

# Stop the integrated system
./stop_integrated_system.sh
```

Note: The integrated system scripts are designed for the full RAG pipeline integration (not yet implemented).

### Accessing the Application

Once running, open your web browser and navigate to:

```
http://localhost:8080
```

You should see the translation interface where you can:
1. Enter or paste text to translate
2. Upload documents (PDF, DOCX)
3. View translation results
4. Access translation history
5. Switch between languages

## API Documentation

### Base URL

```
http://localhost:5002/api
```

### Available Endpoints

#### Health Check

```
GET /api/health
```

Returns the API health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "Translation API",
  "version": "1.0.0"
}
```

#### Translate Text

```
POST /api/translate
```

Translates text from source language to target language.

**Request Body:**
```json
{
  "text": "Hello world",
  "source_language": "en",
  "target_language": "ar",
  "domain": "general"
}
```

**Parameters:**
- `text` (required): Text to translate
- `source_language` (required): Source language code (en, fr)
- `target_language` (required): Target language code (ar)
- `domain` (optional): Translation domain (general, technology, healthcare, legal, etc.)

**Response:**
```json
{
  "translation": "مرحبا بالعالم",
  "source_language": "en",
  "target_language": "ar",
  "service": "deepl"
}
```

#### Detect Language

```
POST /api/detect-language
```

Detects the language of the provided text.

**Request Body:**
```json
{
  "text": "Bonjour le monde"
}
```

**Response:**
```json
{
  "detected_language": "fr",
  "confidence": 0.95
}
```

#### Upload File

```
POST /api/upload
```

Uploads and extracts text from PDF or DOCX files.

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `file`
- Accepted formats: `.pdf`, `.docx`

**Response:**
```json
{
  "text": "Extracted text content...",
  "filename": "document.pdf",
  "page_count": 5
}
```

### Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message description",
  "status": 400
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Evaluation

The project includes a comprehensive evaluation system for measuring translation quality.

### Running Evaluations

Navigate to the Evaluation directory:

```bash
cd Evaluation
```

Install evaluation dependencies:

```bash
pip install -r requirements.txt
```

Ensure the translation API is running (see [Running the Application](#running-the-application)), then run the evaluation:

```bash
python evaluate_translation.py
```

### Evaluation Metrics

The evaluation system computes two standard machine translation metrics:

**BLEU (Bilingual Evaluation Understudy)**
- Measures n-gram precision between hypothesis and reference translations
- Range: 0-100 (higher is better)
- Considers 1-4 gram matches

**CHRF+ (Character n-gram F-score)**
- Character-level metric that handles morphologically rich languages better
- Range: 0-100 (higher is better)
- Particularly suitable for Arabic

### Test Data

The evaluation uses two test datasets:

- `english.csv`: English to Arabic translation pairs
- `french.csv`: French to Arabic translation pairs

Results are automatically saved to the `Results/` directory with timestamps and detailed metrics.

## Future Work

The following features are planned but not yet implemented:

### RAG System (Retrieval-Augmented Generation)

A comprehensive RAG pipeline to enhance translation quality with domain-specific context:

- **Glossary System**: Domain-specific terminology lookup
- **Semantic RAG Retrieval**: Find similar translation examples using vector embeddings
- **Prompt Construction**: Generate optimized prompts for LLM translation

The system architecture includes:
- SQLite FTS5 for glossary lookup
- Qdrant vector database for semantic search
- LaBSE embeddings for cross-lingual similarity
- FastAPI-based microservices on ports 8001-8003

Documentation and implementation guides are available in the `RAG/` directory.

### RLHF System (Reinforcement Learning from Human Feedback)

A planned system to fine-tune translation models based on user feedback:

- Synthetic data generation for training
- Reward model training from human preferences
- PPO (Proximal Policy Optimization) for model optimization
- Interactive inference with continuous learning
- Human preference collection and fine-tuning

Jupyter notebooks with the planned pipeline are available in the `RLHF/` directory.

These systems are designed to work alongside the current translation service to provide:
- Higher quality domain-specific translations
- Continuous improvement through user feedback
- Better handling of specialized terminology
- Context-aware translation suggestions

## Demo Video

A demonstration video showcasing the translation system's features and capabilities is included in the repository.

**Video File**: `RLHF.mp4` (located in the project root directory)

The video covers:
- User interface walkthrough
- Translation functionality (English to Arabic, French to Arabic)
- File upload and processing
- Language detection features
- Translation history management
- API integration demonstration

### Viewing the Demo

To view the demo video:

1. **Local Access**: Navigate to the project root directory and open `RLHF.mp4` with your preferred media player
2. **GitHub**: Download the video file from the repository or view it directly on GitHub if your browser supports it

```bash
# Open the video from command line (Linux)
xdg-open RLHF.mp4

# Open the video from command line (macOS)
open RLHF.mp4
```

## Project Status

**Current Implementation Status:**

- Frontend: Fully implemented
- Backend API: Fully implemented
- Translation Service Integration: Fully implemented (DeepL, OpenRouter)
- File Upload: Fully implemented
- Evaluation System: Fully implemented
- RAG System: Not yet implemented (structure and documentation available)
- RLHF System: Not yet implemented (notebooks and pipeline design available)

## Troubleshooting

### Common Issues

**Port Already in Use**

If you see an error about port 5002 or 8080 being in use:

```bash
# Find and kill the process using the port (Linux/Mac)
lsof -ti:5002 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

**Module Not Found Errors**

If you encounter import errors:

```bash
# Reinstall Python dependencies
cd app
pip install -r requirements.txt --force-reinstall

# Reinstall Node dependencies
rm -rf node_modules package-lock.json
npm install
```

**API Key Issues**

If translations are not working:

1. Verify your `.env` file exists in the `app/` directory
2. Check that your DeepL API key is valid
3. Ensure the `TRANSLATION_SERVICE` variable is set correctly
4. Test your API key at the DeepL website

**CORS Errors**

If you see CORS-related errors in the browser console:

1. Check that `CORS_ORIGINS` in `.env` includes your frontend URL
2. Restart the backend after changing environment variables
3. Clear your browser cache

### Getting Help

For issues not covered here:

1. Check the application logs in the terminal
2. Review the API documentation above
3. Check the individual README files in subdirectories
4. Contact the project maintainers

## License

This project is part of an NLP course project at ENSIA.

## Contributors

- Abdelrahim Anes CHABIRA
- ENSIA 4th Year, NLP Project

## Acknowledgments

- DeepL for providing free translation API access
- The open-source community for the excellent tools and libraries used in this project
