# Context-Specific Machine Translation to Arabic

A modern React frontend with Flask API backend for machine translation, specifically designed for context-aware Arabic translations.

## Features

- **Modern React Frontend**: Built with Vite, TypeScript, and shadcn/ui components
- **Flask API Backend**: RESTful API for translation services
- **Real-time Translation**: API-powered translation with fallback to mock data
- **Multi-language Support**: English, French, and Arabic language detection and translation
- **Responsive Design**: Mobile-first design with Tailwind CSS

## Tech Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **shadcn/ui** component library
- **Tailwind CSS** for styling
- **React Router** for navigation

### Backend
- **Flask** web framework
- **Flask-CORS** for cross-origin requests
- **Python 3.14** with virtual environment

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python 3.14
- npm or yarn

### Installation

1. **Clone the repository**
   ```sh
   git clone <YOUR_GIT_URL>
   cd Front
   ```

2. **Install frontend dependencies**
   ```sh
   npm install
   ```

3. **Set up Python virtual environment and install backend dependencies**
   ```sh
   # The virtual environment is already configured
   # Dependencies are listed in requirements.txt
   # They will be installed automatically when running the API
   ```

### Running the Application

#### Development Mode (Frontend + Backend)

To run both the React frontend and Flask API backend concurrently:

```sh
npm run dev:full
```

This will start:
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:5002

#### Frontend Only

```sh
npm run dev
```

#### Backend API Only

```sh
npm run api
```

### API Endpoints

The Flask API provides the following endpoints:

- `GET /api/health` - Health check
- `POST /api/translate` - Translate text
  ```json
  {
    "text": "Hello world",
    "source_language": "en",
    "target_language": "ar"
  }
  ```
- `POST /api/detect-language` - Detect language of text
  ```json
  {
    "text": "Bonjour le monde"
  }
  ```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Supabase Configuration
VITE_SUPABASE_PROJECT_ID=your_project_id
VITE_SUPABASE_PUBLISHABLE_KEY=your_key
VITE_SUPABASE_URL=your_supabase_url

# Flask API Configuration
FLASK_ENV=development
FLASK_PORT=5002
API_BASE_URL=http://localhost:5002/api
CORS_ORIGINS=http://localhost:5173
```

## Project Structure

```
Front/
├── api/                    # Flask backend
│   ├── __init__.py
│   ├── app.py             # Main Flask application
│   ├── config.py          # Configuration settings
│   └── routes.py          # API endpoints
├── src/                    # React frontend
│   ├── components/        # Reusable UI components
│   ├── pages/            # Page components
│   ├── contexts/         # React contexts
│   ├── hooks/            # Custom hooks
│   ├── lib/              # Utilities and services
│   └── types/            # TypeScript type definitions
├── public/                # Static assets
├── requirements.txt       # Python dependencies
├── package.json          # Node dependencies and scripts
└── .env                  # Environment variables
```

## Development

### Available Scripts

- `npm run dev` - Start frontend development server
- `npm run api` - Start Flask API server
- `npm run dev:full` - Start both frontend and backend
- `npm run build` - Build for production
- `npm run lint` - Run ESLint

### API Integration

The frontend automatically calls the Flask API for translations. If the API is unavailable, it falls back to mock translation data.

## Deployment

### Frontend
```sh
npm run build
```

### Backend
The Flask API can be deployed using gunicorn or similar WSGI server:

```sh
gunicorn api.app:create_app()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and backend
5. Submit a pull request

## License

This project is part of the Context-Specific Machine Translation to Arabic Language research project.
