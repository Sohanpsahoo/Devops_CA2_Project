import os
import pytest
from app import app

@pytest.fixture
def client():
    # Set up a test client for our Flask application
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test that the homepage loads successfully"""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Upload Image' in rv.data or b'classify' in rv.data.lower()

def test_classify_no_image(client):
    """Test that the classify endpoint handles missing images correctly"""
    rv = client.post('/classify', data={})
    assert rv.status_code == 400
    assert b'error' in rv.data
