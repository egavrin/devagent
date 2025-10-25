"""Test cases for user management endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import create_tables

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Setup database before each test."""
    create_tables()


def test_get_users_unauthorized():
    """Test getting users without authentication."""
    response = client.get("/api/v1/users/")
    assert response.status_code == 401


def test_get_users_authorized():
    """Test getting users with valid authentication."""
    # Register and get token
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    token = register_response.json()["access_token"]
    
    # Get users
    response = client.get(
        "/api/v1/users/",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["email"] == "test@example.com"


def test_get_user_by_id():
    """Test getting user by ID."""
    # Register and get token
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    token = register_response.json()["access_token"]
    user_id = register_response.json()["id"]
    
    # Get user by ID
    response = client.get(
        f"/api/v1/users/{user_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["full_name"] == "Test User"


def test_get_user_by_id_not_found():
    """Test getting non-existent user."""
    # Register and get token
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    token = register_response.json()["access_token"]
    
    # Get non-existent user
    response = client.get(
        "/api/v1/users/999",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


def test_update_user_profile():
    """Test updating user profile."""
    # Register and get token
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    token = register_response.json()["access_token"]
    user_id = register_response.json()["id"]
    
    # Update user
    response = client.put(
        f"/api/v1/users/{user_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "full_name": "Updated User",
            "email": "updated@example.com"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "updated@example.com"
    assert data["full_name"] == "Updated User"


def test_delete_user():
    """Test deleting user."""
    # Register and get token
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    token = register_response.json()["access_token"]
    user_id = register_response.json()["id"]
    
    # Delete user
    response = client.delete(
        f"/api/v1/users/{user_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "User deleted successfully"
    
    # Verify user is deleted
    response = client.get(
        f"/api/v1/users/{user_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404
