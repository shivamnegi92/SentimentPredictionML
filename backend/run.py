import uvicorn

if __name__ == "__main__":
    # Launch the FastAPI app
    uvicorn.run(
        "main:app",  # Path to the FastAPI app instance
        host="0.0.0.0",      # Bind to all network interfaces
        port=8000,           # Port number
        reload=True          # Enable hot reload for development
    )
