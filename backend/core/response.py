# backend/core/response.py

from typing import Any, Dict

def success_response(data: Any) -> Dict[str, Any]:
    """
    Utility function to format successful response.
    
    Args:
        data (Any): The data to return in the response.
    
    Returns:
        dict: A dictionary containing the success flag and the data.
    """
    return {"success": True, "data": data}

def error_response(message: str) -> Dict[str, Any]:
    """
    Utility function to format error response.
    
    Args:
        message (str): The error message to return in the response.
    
    Returns:
        dict: A dictionary containing the success flag and the error message.
    """
    return {"success": False, "error": message}
