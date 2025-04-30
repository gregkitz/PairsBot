"""
Models for the web interface.

This module defines the data models used by the web interface,
including the User model for authentication.
"""

import os
import json
import logging
from flask import current_app
from flask_login import UserMixin
from werkzeug.security import check_password_hash

# Configure logging
logger = logging.getLogger(__name__)

class User(UserMixin):
    """User model for authentication."""
    
    def __init__(self, username, password_hash, role='user'):
        self.id = username
        self.username = username
        self.password_hash = password_hash
        self.role = role
    
    def check_password(self, password):
        """
        Check if the provided password matches the user's password.
        
        Parameters:
        -----------
        password : str
            The password to check
        
        Returns:
        --------
        bool
            True if the password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """
        Check if the user is an admin.
        
        Returns:
        --------
        bool
            True if the user is an admin, False otherwise
        """
        return self.role == 'admin'
    
    @staticmethod
    def get(user_id):
        """
        Get a user by ID.
        
        Parameters:
        -----------
        user_id : str
            The user ID (username)
        
        Returns:
        --------
        User or None
            The user object if found, None otherwise
        """
        try:
            users_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'users.json')
            
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
                
                if user_id in users:
                    user_data = users[user_id]
                    return User(
                        username=user_data['username'], 
                        password_hash=user_data['password_hash'],
                        role=user_data.get('role', 'user')
                    )
            
            return None
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            return None
    
    @staticmethod
    def get_all():
        """
        Get all users.
        
        Returns:
        --------
        dict
            Dictionary of all users
        """
        try:
            users_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'users.json')
            
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    return json.load(f)
            
            return {}
        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            return {}
    
    @staticmethod
    def save_user(username, password_hash, role='user'):
        """
        Save a user to the users file.
        
        Parameters:
        -----------
        username : str
            The username
        password_hash : str
            The hashed password
        role : str, optional
            The user role ('user' or 'admin')
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            users_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'users.json')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(users_file), exist_ok=True)
            
            # Load existing users
            users = {}
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
            
            # Add or update user
            users[username] = {
                'username': username,
                'password_hash': password_hash,
                'role': role
            }
            
            # Save users
            with open(users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving user {username}: {str(e)}")
            return False
    
    @staticmethod
    def delete_user(username):
        """
        Delete a user from the users file.
        
        Parameters:
        -----------
        username : str
            The username to delete
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            users_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'users.json')
            
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
                
                if username in users:
                    del users[username]
                    
                    with open(users_file, 'w') as f:
                        json.dump(users, f, indent=2)
                    
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting user {username}: {str(e)}")
            return False 