"""
Routes for the authentication blueprint.

This module defines the routes for user authentication, including
login, logout, and user management.
"""

import logging
from flask import render_template, request, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from . import auth
from ...models import User

# Configure logging
logger = logging.getLogger(__name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    # Check if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    # Handle login form submission
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate input
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('auth/login.html')
        
        # Get user
        user = User.get(username)
        
        # Check if user exists and password is correct
        if user and user.check_password(password):
            login_user(user)
            logger.info(f"User {username} logged in")
            
            # Redirect to requested URL or dashboard
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard.index'))
        else:
            flash('Invalid username or password', 'danger')
    
    # Render login form
    return render_template('auth/login.html')

@auth.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logger.info(f"User {current_user.username} logged out")
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

@auth.route('/users')
@login_required
def users():
    """Display user management page."""
    # Only allow admin users
    if not current_user.is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get all users
    users = User.get_all()
    
    return render_template('auth/users.html', users=users)

@auth.route('/users/add', methods=['GET', 'POST'])
@login_required
def add_user():
    """Add a new user."""
    # Only allow admin users
    if not current_user.is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Handle form submission
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role', 'user')
        
        # Validate input
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('auth/add_user.html')
        
        # Check if user already exists
        existing_user = User.get(username)
        if existing_user:
            flash(f"User '{username}' already exists", 'danger')
            return render_template('auth/add_user.html')
        
        # Create user
        password_hash = generate_password_hash(password)
        if User.save_user(username, password_hash, role):
            logger.info(f"User {username} created by {current_user.username}")
            flash(f"User '{username}' created successfully", 'success')
            return redirect(url_for('auth.users'))
        else:
            flash("Failed to create user", 'danger')
    
    # Render add user form
    return render_template('auth/add_user.html')

@auth.route('/users/edit/<username>', methods=['GET', 'POST'])
@login_required
def edit_user(username):
    """Edit an existing user."""
    # Only allow admin users or the user themselves
    if not current_user.is_admin() and current_user.username != username:
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get user
    user = User.get(username)
    if not user:
        flash(f"User '{username}' not found", 'danger')
        return redirect(url_for('auth.users'))
    
    # Handle form submission
    if request.method == 'POST':
        password = request.form.get('password')
        role = request.form.get('role')
        
        # Only admins can change roles
        if role and current_user.is_admin():
            # Keep at least one admin
            users = User.get_all()
            admin_count = sum(1 for u in users.values() if u.get('role') == 'admin')
            
            if user.role == 'admin' and role != 'admin' and admin_count <= 1:
                flash('Cannot remove the last admin user', 'danger')
                return render_template('auth/edit_user.html', user=user)
        else:
            # Use existing role if not admin or no role provided
            role = user.role
        
        # Update user
        if password:
            # Update password if provided
            password_hash = generate_password_hash(password)
        else:
            # Keep existing password if not provided
            password_hash = user.password_hash
        
        if User.save_user(username, password_hash, role):
            logger.info(f"User {username} updated by {current_user.username}")
            flash(f"User '{username}' updated successfully", 'success')
            
            # Redirect to users page if admin, or dashboard if regular user
            if current_user.is_admin():
                return redirect(url_for('auth.users'))
            else:
                return redirect(url_for('dashboard.index'))
        else:
            flash("Failed to update user", 'danger')
    
    # Render edit user form
    return render_template('auth/edit_user.html', user=user)

@auth.route('/users/delete/<username>', methods=['POST'])
@login_required
def delete_user(username):
    """Delete a user."""
    # Only allow admin users
    if not current_user.is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get user
    user = User.get(username)
    if not user:
        flash(f"User '{username}' not found", 'danger')
        return redirect(url_for('auth.users'))
    
    # Prevent deleting self
    if username == current_user.username:
        flash('You cannot delete your own account', 'danger')
        return redirect(url_for('auth.users'))
    
    # Prevent deleting last admin
    if user.role == 'admin':
        users = User.get_all()
        admin_count = sum(1 for u in users.values() if u.get('role') == 'admin')
        
        if admin_count <= 1:
            flash('Cannot delete the last admin user', 'danger')
            return redirect(url_for('auth.users'))
    
    # Delete user
    if User.delete_user(username):
        logger.info(f"User {username} deleted by {current_user.username}")
        flash(f"User '{username}' deleted successfully", 'success')
    else:
        flash(f"Failed to delete user '{username}'", 'danger')
    
    return redirect(url_for('auth.users'))

@auth.route('/profile')
@login_required
def profile():
    """Display user profile."""
    return render_template('auth/profile.html') 