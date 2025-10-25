# REST API with Authentication

A complete, production-ready REST API built with FastAPI featuring user authentication, comprehensive testing, and security best practices.

## Features

- ✅ **User Authentication**: JWT-based authentication with registration and login
- ✅ **User Management**: CRUD operations for user profiles
- ✅ **Security**: Password hashing, rate limiting, security headers
- ✅ **Testing**: Comprehensive test suite with pytest
- ✅ **Documentation**: Auto-generated OpenAPI documentation
- ✅ **Database**: SQLAlchemy ORM with SQLite (configurable)
- ✅ **Environment Configuration**: Environment variables for all settings

## Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rest_api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run the application:
```bash
python -m app.main
```

5. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user
- `GET /api/v1/auth/me` - Get current user profile

### Users

- `GET /api/v1/users/` - Get all users (authenticated)
- `GET /api/v1/users/{id}` - Get user by ID (authenticated)
- `PUT /api/v1/users/{id}` - Update user profile (authenticated)
- `DELETE /api/v1/users/{id}` - Delete user (authenticated)

### System

- `GET /` - API information
- `GET /health` - Health check

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=app --cov-report=html
```

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt for secure password storage
- **Rate Limiting**: Prevents API abuse (100 requests/minute)
- **Security Headers**: XSS protection, content security policy
- **CORS**: Configured for frontend applications
- **Input Validation**: Pydantic models for request validation

## Configuration

Environment variables in `.env`:

```bash
# Database
DATABASE_URL=sqlite:///./app.db

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Security
BCRYPT_ROUNDS=12

# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

## Development

### Project Structure

```
rest_api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database configuration
│   ├── models/              # SQLAlchemy models
│   ├── routers/             # API route handlers
│   ├── utils/               # Utility functions
│   └── middleware/          # Security middleware
├── tests/                   # Test suite
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Adding New Features

1. Create models in `app/models/`
2. Add routers in `app/routers/`
3. Write tests in `tests/`
4. Update documentation

## Deployment

For production deployment:

1. Set `DEBUG=False` in environment
2. Use a production database (PostgreSQL recommended)
3. Set strong `SECRET_KEY`
4. Use HTTPS with reverse proxy (nginx)
5. Configure proper CORS origins

## License

MIT License
