"""
JSON Utility Functions

This module provides utility functions for JSON serialization,
particularly focusing on handling datetime objects.
"""

import json
import datetime
import logging

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that properly handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return super().default(obj)

def patch_json_encoder():
    """
    Monkey patches the default JSON encoder to handle datetime objects.
    This allows datetime objects to be serialized properly throughout the application.
    
    Returns:
        bool: True if patching was successful, False otherwise
    """
    try:
        # Save references to original functions
        original_dumps = json.dumps
        original_dump = json.dump
        
        # Define patched versions
        def patched_dumps(*args, **kwargs):
            if 'cls' not in kwargs:
                kwargs['cls'] = DateTimeEncoder
            return original_dumps(*args, **kwargs)
        
        def patched_dump(*args, **kwargs):
            if 'cls' not in kwargs:
                kwargs['cls'] = DateTimeEncoder
            return original_dump(*args, **kwargs)
        
        # Apply the patches
        json.dumps = patched_dumps
        json.dump = patched_dump
        
        logger.info("Successfully patched JSON encoder to handle datetime objects")
        return True
    except Exception as e:
        logger.error(f"Error patching JSON encoder: {e}")
        return False

def parse_datetime(datetime_str):
    """
    Parse an ISO formatted datetime string back into a datetime object.
    
    Args:
        datetime_str (str): ISO formatted datetime string
        
    Returns:
        datetime.datetime: Parsed datetime object
    """
    try:
        return datetime.datetime.fromisoformat(datetime_str)
    except ValueError:
        # Handle cases where the string might not be in ISO format
        try:
            # Try with multiple formats
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Could not parse datetime string: {datetime_str}")
        except Exception as e:
            logger.error(f"Error parsing datetime: {e}")
            raise 