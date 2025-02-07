# Sentiment Prediction API

## CI-CD Failure: Missing Model and Tokenizer in `model_store`

**Important:** Our CI-CD pipeline is currently failing because the model and tokenizer required by the FastAPI backend are missing from the `model_store` folder.  This is due to file size limitations.

The model and tokenizer are generated within the notebook and saved in the `final_tokenizer` and `final_model` folders, respectively.  Currently, these files need to be *manually* moved to the `model_store` folder in the backend repository to resolve the CI-CD failure.  This manual step is a temporary workaround.


This project provides an API for sentiment analysis using a pre-trained model.

## Endpoints

- `POST /predict/single`: Get sentiment prediction for a single text.
- `POST /predict/batch`: Get sentiment predictions for a list of texts.
- `GET /health`: Check if the API is healthy.



### Detailed Documentation: **Sentiment Prediction FastAPI Backend**

This document explains the architecture and implementation of a scalable and modular **Sentiment Prediction Backend** built using **FastAPI**. The backend serves a machine learning model for both single and batch sentiment prediction using a pre-trained **BERT** model. The backend is structured in an enterprise-level manner, following modular programming principles to ensure ease of maintainability, extensibility, and scalability.

---

### **Folder Structure Overview**

Here is the folder structure for the project:

```
backend/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── batch_predict.py
│   │   │   ├── predict.py
│   │   │   ├── health.py
├── services/
│   ├── __init__.py
│   └── inference.py
├── models/
│   ├── __init__.py
│   └── sentiment_model.py
├── utils/
│   └── __init__.py
├── config/
│   ├── __init__.py
│   └── config.py
├── models_store/
├── core/
│   ├── __init__.py
│   ├── logger.py
│   ├── response.py
│   └── health.py
├── main.py
tests/
├── __init__.py
├── test_api.py
├── test_services.py
└── test_inference.py
.env
requirements.txt
README.md
run.py

```

---

### **Fundamental Understanding of the Backend Codebase Setup**

This document provides a conceptual breakdown of the architecture of the **Sentiment Prediction Backend**. It explains the role of each component and the reasoning behind this setup. The backend follows modular, layered architecture to promote scalability, maintainability, and separation of concerns.

### **1. Overview: Modular Architecture**

The **FastAPI Backend** is organized using a **layered architecture** with a **modular structure**. This setup helps separate different concerns, such as API handling, business logic, model inference, response formatting, and configuration management.

In this architecture:

- **Layer 1**: **API Layer**: Handles HTTP requests, routing, and validation.
- **Layer 2**: **Service Layer**: Contains business logic such as model inference, data processing, and transformation.
- **Layer 3**: **Model Layer**: Loads, processes, and serves the machine learning models.
- **Layer 4**: **Core Layer**: Contains utilities such as logging, response formatting, and configuration management.

The **empty `__init__.py` files** in each folder are not just placeholders. They serve specific purposes that enhance the modularity and scalability of the codebase. Below, I will break down the different layers, their roles, and the reasoning behind using `__init__.py` files.

---

### **2. Layered Architecture Breakdown**

### **Layer 1: API Layer**

The API layer is responsible for handling incoming HTTP requests and sending responses back to the client. FastAPI's routing mechanism is used to define and manage the API endpoints.

### **Files Involved**:

- `backend/api/v1/endpoints/predict.py`
- `backend/api/v1/endpoints/batch_predict.py`
- `backend/api/v1/endpoints/health.py`

### **Role**:

- **Route Definition**: Each file in this folder defines a set of API routes (endpoints). For example, `predict.py` handles the route for single text prediction (`/predict/`), while `batch_predict.py` handles the batch prediction route (`/batch/`).
- **Request Handling**: These files define the functions that handle the incoming HTTP requests, validate the data, and return appropriate responses.
- **Modularization**: Keeping each endpoint in a separate file ensures the API is modular and scalable. For instance, you can easily add new endpoints for other functionalities (e.g., sentiment analysis of images, user authentication, etc.).

### **Example**:

In `predict.py`:

```python
from fastapi import APIRouter  # Import API router class from FastAPI

from backend.services.inference import predict

from backend.api.v1.schemas.prediction import PredictionRequest

from backend.core.response import success_response

# Initialize a router object
router = APIRouter()

# @router: This part is a decorator that's used in FastAPI for defining API endpoints (routes) within a FastAPI application. It associates the function that follows it with the router object (router in this case).

@router.post("")
async def predict_single(request: PredictionRequest):
    """
    This function takes a PredictionRequest object as input,
    which contains the text data to be predicted on.
    It calls the predict function from the inference service to make the prediction
    and returns a success response with the prediction result.
    """
    result = await predict(request.text)  # Call predict function from inference service and wait for the result
    return success_response({"prediction": result})  # Return a success response with the prediction
```

Here:

- `APIRouter()` creates a new router to define API routes.
- The `@router.post("")` decorator maps the HTTP POST request to the `predict_single` function. This function calls the `predict()` function from the service layer, gets the result, and then formats the response using `success_response()`.

The purpose of this decorator is to create an endpoint within theFastAPI application that can receive POST requests from clients. When a client sends a POST request to the root path of the API, the function decorated with `@router.post("")` will be invoked to handle the request.

**`async` keyword:**

- **`async`**: Yes, `async` is a keyword in Python that's used to define asynchronous functions. Asynchronous functions are functions that can be paused and resumed without blocking the execution of the main program. This is particularly useful for I/O-bound operations (like network requests or database interactions) where the program might need to wait for a response before continuing.

### **Why `__init__.py` is used here**:

- **To Make it a Python Package**: The `__init__.py` file is necessary for Python to treat the folder as a package. Without it, Python cannot import files from that folder.
- **Allows Easy Importing**: With the `__init__.py` file in place, you can import routes and modules from these subdirectories easily, like `from backend.api.v1.endpoints import predict`, without worrying about relative imports.

---

### **Layer 2: Service Layer**

The service layer contains the core business logic of the application. This layer orchestrates the flow of data, invoking the model for predictions, handling inputs/outputs, and managing intermediate steps.

### **Files Involved**:

- `backend/services/inference.py`

### **Role**:

- **Business Logic**: This layer deals with the actual business processing — for example, receiving text data, tokenizing it, invoking the model, and processing the output. It acts as a bridge between the API layer and the model layer.
- **Model Invocation**: The service layer calls the functions of the model layer (i.e., `SentimentModel` in `backend/models/sentiment_model.py`) to make predictions.

### **Example**:

In `inference.py`:

```python
"""
This module contains functions and classes for handling inference
using the SentimentModel.
"""

from backend.models.sentiment_model import SentimentModel
from typing import List

# Load the sentiment model
model = SentimentModel()

async def predict(text: str) -> str:
    """Predict the output based on the input text.

    Args:
        text (str): The input text for prediction.

    Returns:
        str: The predicted output.
    """
    return model.predict(text)

async def batch_predict(texts: List[str]) -> List[str]:
    """
    Batch prediction function that processes a list of texts and returns predictions.
    
    Args:
        texts (List[str]): A list of texts for sentiment prediction.
    
    Returns:
        List[str]: A list of predictions (e.g., "Positive", "Negative").
    """
    return model.batch_predict(texts)

```

Here:

- The `predict` function in `inference.py` calls the `predict()` method of the `SentimentModel` to get the sentiment prediction.

### **Why `__init__.py` is used here**:

- **Modularization**: The `__init__.py` allows this folder to be a package, making it easier to import and use the business logic in the service layer (e.g., `from backend.services.inference import predict`).

---

### **Layer 3: Model Layer**

The model layer contains the code related to the machine learning model. This includes loading the trained model, running predictions, and returning the result.

### **Files Involved**:

- `backend/models/sentiment_model.py`

### **Role**:

- **Model Loading and Inference**: This layer is responsible for loading the pre-trained model (e.g., BERT) and tokenizer, processing the input text, and returning the sentiment prediction (positive or negative).

### **Example**:

In `sentiment_model.py`:

```python
from typing import List  # <-- Import List here

import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentModel:
    def __init__(self):
        # Load the model and tokenizer from models_store
        self.model = BertForSequenceClassification.from_pretrained('backend/models_store/final_model')  # Load the fine-tuned model
        self.model.eval()  # Set model to evaluation mode
        
        # Load tokenizer from the saved location
        self.tokenizer = BertTokenizer.from_pretrained('backend/models_store/final_tokenizer')  # Load the tokenizer

    def batch_predict(self, texts: List[str]) -> List[str]:
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class labels
        predictions = outputs.logits.argmax(dim=1)  # Get the index of the highest probability (class prediction)
        
        # Return predictions as a list of strings (e.g., "Positive", "Negative")
        return [str(pred.item()) for pred in predictions]

    def predict(self, text: str) -> str:
        # For a single text, we can just call batch_predict with one element
        return self.batch_predict([text])[0]  # Return first prediction for single text

```

Here:

- The `SentimentModel` class loads the model and tokenizer.
- The `predict()` method processes the input text and runs inference.

### **Why `__init__.py` is used here**:

- **Modularization**: The `__init__.py` allows the model-related code to be encapsulated in a module that can easily be imported into the service layer (`from backend.models.sentiment_model import SentimentModel`).

---

### **Layer 4: Core Layer**

The core layer contains utility functions that support other layers, such as response formatting, logging, configuration, and health checks.

### **Files Involved**:

- `backend/core/response.py`
- `backend/core/logger.py`
- `backend/core/health.py`

### **Role**:

- **Response Formatting**: Functions like `success_response()` and `error_response()` in `response.py` format the response that will be sent to the client. This ensures a consistent structure for all responses.
- **Logging**: The `logger.py` file contains the logging setup that logs important events and errors in the application.
- **Health Check**: The `health.py` file contains a simple endpoint for checking whether the service is running correctly.

### **Example**:

In `response.py`:

```python
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

```

Here:

- `success_response()` is used to wrap the prediction result in a consistent structure that contains a `success` field.

### **Why `__init__.py` is used here**:

- **Package Initialization**: The `__init__.py` ensures the folder is treated as a package, which allows easier import of utility functions in other layers, such as `from backend.core.response import success_response`.

---

### **Conclusion**

- **Modular Design**: The project is structured into layers to ensure separation of concerns and scalability. Each layer has a clear role, and the code components are organized to keep different concerns separated. For example, API handling, business logic, and model inference are all separated into different layers.
- **Use of `__init__.py`**: The use of `__init__.py` files in each folder is essential for treating the folders as Python packages. This allows the folders to be imported easily into other parts of the application, maintaining modularity and scalability. These files enable clean and maintainable code that can be easily extended as the project grows.
- **Scalability**: This setup allows you to easily extend the application. For example, adding new prediction models, adding more endpoints, or integrating additional services would not require major changes to the existing codebase. Instead, new components can be added to the appropriate layers (API, services, models, etc.).

### **Role of Each Folder and File**

### **1. `backend/`**:

This is the root folder that contains all the backend code. All logic related to the FastAPI backend resides here, including APIs, services, models, utility functions, configuration, and core components.

---

### **2. `api/`**:

This folder contains the API layer, specifically the endpoint definitions that expose the FastAPI routes. It follows versioning for flexibility in case of future API changes.

- **`v1/`**: This folder contains version 1 of the API. As the application evolves, new versions can be created under a separate folder (e.g., `v2/`).
    - **`endpoints/`**: This folder contains the actual route definitions for various endpoints like prediction and health check.
    - **`batch_predict.py`**: This file defines the endpoint for batch prediction. It handles requests that include multiple texts and returns predictions for all texts.
    - **`predict.py`**: This file defines the endpoint for single-text prediction. It handles requests with a single text and returns a single prediction.
    - **`health.py`**: This file defines a simple health check endpoint to verify the status of the API server.

---

### **3. `services/`**:

This folder contains business logic and service layer code that is invoked by the API routes. These services can contain more complex processing and business logic.

- **`inference.py`**: Contains the logic for invoking the model for predictions. The functions here handle the interaction with the machine learning model, including preprocessing and postprocessing.

---

### **4. `models/`**:

This folder contains the machine learning models and their related code.

- **`sentiment_model.py`**: Contains the `SentimentModel` class, which is responsible for loading the model and tokenizer, and making predictions.

---

### **5. `utils/`**:

This folder contains utility functions that can be reused throughout the application.

---

### **6. `config/`**:

This folder contains configuration files for the application.

- **`config.py`**: This file holds application settings, such as model paths, server configurations, etc.

---

### **7. `models_store/`**:

This folder stores the actual trained machine learning models and tokenizers.

- **`final_model/`**: Contains the trained machine learning model (e.g., `model.pth`).
- **`final_tokenizer/`**: Contains the tokenizer used to process text data before passing it into the model.

---

### **8. `core/`**:

This folder contains core utility files that handle logging, responses, and health checks.

- **`logger.py`**: Contains logging setup, helping monitor the application during runtime.
- **`response.py`**: Contains utility functions for formatting successful or error responses.
- **`health.py`**: Contains the health check logic for the application, ensuring the system is running correctly.

---

### **9. `main.py`**:

The entry point for the FastAPI application. This file sets up the FastAPI app, includes routers for the API endpoints, and configures logging.

- It imports the routers defined in the `api/v1/endpoints/` and includes them in the FastAPI application with appropriate prefixes (e.g., `/predict`, `/batch`, `/health`).

---

### **10. `tests/`**:

This folder contains test cases for various components of the application.

- **`test_api.py`**: Contains tests for the API layer, ensuring that the routes work correctly.
- **`test_services.py`**: Contains tests for the service layer, verifying that the logic inside services like `inference.py` is correct.
- **`test_inference.py`**: Contains tests for the model inference code, ensuring that predictions are made accurately.

---

### **11. `run.py`**:

This is the script to run the FastAPI application. It ensures that the application is launched using `uvicorn` and the server is up and running.

---

### **12. `requirements.txt`**:

This file contains all the Python dependencies for the project, such as `FastAPI`, `uvicorn`, `transformers`, `torch`, etc. These dependencies are required for running the API and the model inference.

---

### **13. `.env`**:

This file holds environment-specific configurations (e.g., API keys, model paths, etc.). This file can be read using Python's `dotenv` package.

---

### **14. `README.md`**:

This file provides a high-level overview of the project. It typically includes instructions on how to set up and run the application.

---

### **Code Flow and Execution**

1. **Starting the Server**:
To run the application, go to the root folder which is the base folder and you start the FastAPI server using the `run.py` file or directly via `uvicorn`. When the server is started, FastAPI reads the routers and includes them based on their definitions.
    
    ```bash
    uvicorn backend.main:app --reload
    
    ```
    
    - **FastAPI Application Initialization**: The FastAPI instance is created in `main.py` using `FastAPI()`.
    - **Router Inclusion**: The `main.py` file includes the routes defined in the `api/v1/endpoints` module. These routes map the URL paths (`/predict`, `/batch`, `/health`) to the functions defined in `predict.py`, `batch_predict.py`, and `health.py`.

Using [run.py](http://run.py) in cmd

```bash
python  run.py

```

1. **API Requests**:
Once the server is running, you can send HTTP requests to the API:
    - **Single Text Prediction**:
        - Send a `POST` request to `/predict/` with the body `{ "text": "The movie was amazing!" }`.
        - This request hits the `predict_single()` function in `predict.py`.
        - The request is processed by the `predict()` function from `services/inference.py`, which calls the `SentimentModel` for prediction and returns the result as a response.
    - **Batch Text Prediction**:
        - Send a `POST` request to `/batch/` with the body:
            
            ```json
            [
              { "text": "The movie was amazing!" },
              { "text": "I did not enjoy the film." }
            ]
            
            ```
            
        - This request hits the `batch_predict_single()` function in `batch_predict.py`.
        - It processes the list of texts using the `batch_predict()` function from `services/inference.py` and returns the predictions for each text.
2. **Model Inference**:
The model inference is done using the `SentimentModel` class located in `models/sentiment_model.py`. This class is responsible for loading the BERT model and tokenizer from the `models_store` folder, tokenizing the input text, and performing the forward pass to get predictions.
    - **SentimentModel**:
        - Loads the model and tokenizer in the `__init__()` method.
        - The `predict()` method handles single-text predictions by calling `batch_predict()`.
        - The `batch_predict()` method handles multiple texts, tokenizes them, and returns the class predictions for each text.
3. **Logging**:
The `logger.py` module is responsible for logging important events during the execution of the server. It helps in debugging and monitoring the server’s status.

---

### **Execution Flow in Detail**:

1. **API Request**:
    - A client (e.g., Postman) sends an HTTP `POST` request to the `/predict/` or `/batch/` endpoint.
    - FastAPI receives the request and forwards it to the appropriate function based on the endpoint.
2. **Prediction**:
    - The request is processed in the service layer, where the model is invoked using the `SentimentModel` class.
    - The `predict()` or `batch_predict()` methods are called on the `SentimentModel` class to generate predictions.
3. **Response**:
    - The result from the model inference

is then returned as a response. The response is formatted using the `success_response()` function from `response.py`.

1. **Logging**:
    - All important events are logged in `logger.py`, ensuring you can track the execution.

---

---

### **Explanation of the Folder Structure**

1. **`backend/`** – This is the root module for the backend, and it contains everything related to the backend API.
    - **`__init__.py`** – This file marks the folder as a Python package, making it a module. You will have this in all subfolders to ensure modularity.
2. **`api/`** – Contains all the API route definitions (FastAPI routes).
    - **`v1/`** – A versioned folder (API version 1).
        - **`__init__.py`** – Marks the `v1` folder as a Python module.
        - **`predict.py`** – Endpoint for single inference.
        - **`batch_predict.py`** – Endpoint for batch inference.
        - **`health.py`** – Health check endpoint.
    - **`v2/`** (optional) – For future versions of the API (e.g., v2).
        - You can add the same routes here when you need versioned APIs.
3. **`services/`** – Contains business logic, including the logic for model inference, batch processing, etc.
    - **`inference.py`** – Handles the inference logic for both single and batch predictions.
4. **`models/`** – Contains Pydantic models for request and response validation.
    - **`inference.py`** – Contains request models (for single and batch prediction) and response models (e.g., prediction output).
5. **`utils/`** – Contains utility functions (e.g., tokenization, preprocessing).
6. **`config/`** – Contains configuration files and settings (e.g., environment variables, model paths).
    - **`config.py`** – Stores configuration for paths, environment variables, app name, etc.
7. **`models_store/`** – Folder to store themodel and tokenizer.
8. **`core/`** – Core utilities and foundational components for the application.
    - **`logger.py`** – Logger setup.
    - **`device.py`** – Device management for selecting GPU or CPU.
    - **`response.py`** – Common response structures.
    - **`health.py`** – Health check-related logic.
9. **`main.py`** – The entry point for the FastAPI application.

---

### **Next Steps: File Creation Order**

Here’s the order of file creation for better understanding:

1. **`backend/__init__.py`** – Mark the backend directory as a module.
2. **`backend/api/__init__.py`** – Mark the `api` folder as a module.
3. **`backend/api/v1/__init__.py`** – Mark the `v1` folder as a module.
4. **`backend/api/v1/predict.py`** – Define the routes for single inference.
5. **`backend/api/v1/batch_predict.py`** – Define the routes for batch inference.
6. **`backend/api/v1/health.py`** – Define the health check route.
7. **`backend/services/__init__.py`** – Mark the `services` folder as a module.
8. **`backend/services/inference.py`** – Implement business logic for inference.
9. **`backend/models/__init__.py`** – Mark the `models` folder as a module.
10. **`backend/models/inference.py`** – Define Pydantic models for request and response.
11. **`backend/utils/__init__.py`** – Mark the `utils` folder as a module.
12. **`backend/config/__init__.py`** – Mark the `config` folder as a module.
13. **`backend/config/config.py`** – Define the configuration file (paths, settings).
14. **`backend/core/__init__.py`** – Mark the `core` folder as a module.
15. **`backend/core/logger.py`** – Implement logging.
16. **`backend/core/device.py`** – Implement device selection for model inference (CPU/GPU).
17. **`backend/core/response.py`** – Define common response formats.
18. **`backend/core/health.py`** – Implement health check logic.
19. **`backend/main.py`** – Set up FastAPI app and include routes.
20. **`tests/__init__.py`** – Mark the tests folder as a module.
21. **`tests/test_api.py`** – Write unit tests for API routes.
22. **`tests/test_services.py`** – Write unit tests for the business logic (inference).
23. **`tests/test_inference.py`** – Write unit tests for inference logic.

---

### 

```python
# backend/main.py
from fastapi import FastAPI
from backend.api.v1 import predict, batch_predict, health
from backend.api.v2 import batch_predict as batch_predict_v2

app = FastAPI()

# Include v1 routes
app.include_router(predict.router, prefix="/v1")
app.include_router(batch_predict.router, prefix="/v1")

```

To create the `backend` module with code in the folder structure, let's go step-by-step, explaining each file and how it connects with others. Each file's purpose will be detailed to provide contextual understanding.

---

## **Step 1: `backend/main.py`**

### **Purpose**

This is the entry point of the application. It starts the FastAPI server and includes middleware, exception handling, and routing.

### **Code**

```python
# backend/main.py
from fastapi import FastAPI
from backend.api.v1.endpoints import predict, batch_predict, health
from backend.core.logger import setup_logging

# Initialize logging
setup_logging()

# Create FastAPI app
app = FastAPI(title="Sentiment Prediction API", version="1.0.0")

# Log inclusion for debugging
print("Including batch_predict router")

# Include routers with correct prefixes
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(batch_predict.router, prefix="/batch", tags=["Batch Prediction"])
app.include_router(health.router, prefix="/health", tags=["Health"])

```

### **Context**

- Initializes the FastAPI application.
- Includes routes from `api/v1` files (`predict`, `batch_predict`, `health`).
- Calls the `init_logging` function from `core.logger`.

---

## **Step 2: `backend/core/logger.py`**

### **Purpose**

Sets up logging for the application to ensure errors, warnings, and information are logged consistently.

### **Code**

```python
import logging

def setup_logging():
    logger = logging.getLogger("uvicorn")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

```

### **Context**

- Centralized logging configuration.
- Used by other parts of the application to log messages.

**Logging Levels:**

The Python logging module defines different logging levels that categorize the severity of messages:

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
- `INFO`: Confirmation that things are working as expected.
- `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., disk space low).
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
- `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running.

By setting the logger level to `INFO`, you're controlling which messages are actually logged. You can adjust this level to suit theneeds:

- Set it to `DEBUG` to see all messages, including debugging information.
- Set it to `WARNING` to see only warnings and errors.

**Viewing Logged Messages:**

Once you've configured logging and set the desired level, you can view the logged messages by running theapplication. The messages will be printed to the console (standard output).

sample logger output gets printed when application is started:

```bash
INFO:     Will watch for changes in these directories: ['C:\\Users\\Documents\\Data Science Work\\Github\\SentimentPredictionML']
INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000/) (Press CTRL+C to quit)
INFO:     Started reloader process [1020668] using WatchFiles
```

---

## **Step 3: `backend/core/device.py`**

### **Purpose**

Determines the device to use for inference (CPU or GPU).

### **Code**

```python
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

### **Context**

- Other modules (e.g., `services.inference`) use this to decide whether the model should run on GPU or CPU.

---

## **Step 4: `backend/core/response.py`**

### **Purpose**

Standardizes API responses across the application.

### **Code**

```python
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

```

### **Context**

- Used by API endpoints to return consistent responses to the client.

---

## **Step 5: `backend/config/config.py`**

### **Purpose**

Centralized configuration for the application, including model paths and environment variables.

### **Code**

```python
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "backend/models_store/final_model")  # Default value is set to path in backend
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "backend/models_store/final_tokenizer")  # Tokenizer path

```

### **Context**

- Allows easy modification of paths and settings without changing code logic.
- Uses `.env` for environment variables.

---

## **Step 6: `backend/models/sentiment_model.py`**

### **Purpose**

Loads the model and tokenizer for prediction tasks.

### **Code**

```python
"""
This module contains the sentiment analysis model using BERT.
"""
from typing import List
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from backend.config import MODEL_PATH, TOKENIZER_PATH  # Import config values

class SentimentModel:
    def __init__(self):
        # Load the model and tokenizer using paths from config
        self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH)  # Load the fine-tuned model
        self.model.eval()  # Set model to evaluation mode
        
        # Load tokenizer from the saved location
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)  # Load the tokenizer

    def batch_predict(self, texts: List[str]) -> List[str]:
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class labels
        predictions = outputs.logits.argmax(dim=1)  # Get the index of the highest probability (class prediction)
        
        # Return predictions as a list of strings (e.g., "Positive", "Negative")
        return [str(pred.item()) for pred in predictions]

    def predict(self, text: str) -> str:
        # For a single text, we can just call batch_predict with one element
        return self.batch_predict([text])[0]  # Return first prediction for single text

```

### **Context**

- **Model Loading**: The `SentimentModel` class loads a fine-tuned BERT model and tokenizer from specified paths in the configuration file.
    - The tokenizer processes the list of input texts:
    - `padding=True` ensures all input texts are padded to the same length.
    - `truncation=True` ensures any text longer than the model's maximum input length is truncated.
    - `return_tensors="pt"` ensures the output is a PyTorch tensor, which is required for model inference.
- **Inference**: The model can make predictions on both individual texts and batches of texts. It uses PyTorch for model inference and the Hugging Face `transformers` library for tokenization and model management.
- **Batch and Single Prediction**: You can use `batch_predict` to predict sentiment for multiple texts simultaneously, and `predict` to predict sentiment for a single text.
- Used by `services.inference`.

---

## **Step 7: `backend/services/inference.py`**

### **Purpose**

Performs inference using the loaded model and tokenizer.

### **Code**

```python
"""
This module contains functions and classes for handling inference
using the SentimentModel.
"""

from backend.models.sentiment_model import SentimentModel
from typing import List

# Load the sentiment model
model = SentimentModel()

async def predict(text: str) -> str:
    """Predict the output based on the input text.

    Args:
        text (str): The input text for prediction.

    Returns:
        str: The predicted output.
    """
    return model.predict(text)

async def batch_predict(texts: List[str]) -> List[str]:
    """
    Batch prediction function that processes a list of texts and returns predictions.
    
    Args:
        texts (List[str]): A list of texts for sentiment prediction.
    
    Returns:
        List[str]: A list of predictions (e.g., "Positive", "Negative").
    """
    return model.batch_predict(texts)

```

### **Context**

- Encapsulates the logic for tokenization and inference.
- Decouples model handling from API logic.

---

## **Step 8: `backend/api/v1/predict.py`**

### **Purpose**

Defines the API endpoint for single inference.

### **Code**

```python
"""
This module contains the API endpoints for making predictions.
"""
from fastapi import APIRouter  # Import API router class from FastAPI
from backend.services.inference import predict
from backend.api.v1.schemas.prediction import PredictionRequest
from backend.core.response import success_response

# Initialize a router object
router = APIRouter()  # type: ignore
"""
This router handles API endpoints for predictions.
"""

# @router: This part is a decorator that's used in FastAPI for defining API
# endpoints (routes) within a FastAPI application. It associates the function that
# follows it with the router object (router in this case).

@router.post("")
async def predict_single(request: PredictionRequest):
    """
    This function takes a PredictionRequest object as input,
    which contains the text data to be predicted on.
    It calls the predict function from the inference service to make the prediction
    and returns a success response with the prediction result.
    """
    result = await predict(request.text)  # Call predict function from inference service and wait for the result
    return success_response({"prediction": result})  # Return a success response with the prediction
```

### **Context**

- Handles API requests for single prediction.
- Uses `InferenceService` to make predictions.

---

## **Step 9: `backend/api/v1/batch_predict.py`**

### **Purpose**

Defines the API endpoint for batch inference.

### **Code**

```python
# backend/api/v1/endpoints/batch_predict.py
from fastapi import APIRouter
from backend.services.inference import batch_predict
from backend.api.v1.schemas.prediction import PredictionRequest
from backend.core.response import success_response, error_response
from typing import List

router = APIRouter()

@router.post("")
async def batch_predict_single(request: List[PredictionRequest]):
    try:
        # Extract text data from the request list of PredictionRequest objects
        texts = [item.text for item in request]
        
        # Use the batch prediction method
        predictions = await batch_predict(texts)
        
        return success_response({"predictions": predictions})
    except Exception as e:
        return error_response(str(e))
```

### **Context**

- Handles API requests for batch predictions.

---

## **Step 10: `backend/api/v1/health.py`**

### **Purpose**

Provides a health check endpoint for the API.

### **Code**

```python
# backend/api/v1/health.py

from fastapi import APIRouter

# Define a FastAPI APIRouter instance
router = APIRouter()

# Define a simple health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "ok"}

```

### **Context**

- Used for monitoring and confirming the service is operational.

## **Step 11: `backend/v1/schemas/prediction.py`**

```python
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]

```

The code defines **Pydantic schemas** for handling data validation and serialization in theFastAPI application. Let’s break down what these schemas are doing and why they are used in theFastAPI backend.

### Pydantic and FastAPI

**Pydantic** is a data validation and parsing library that integrates seamlessly with FastAPI. It's used to:

- **Validate incoming data** (e.g., whether the expected fields are present and have the correct data types).
- **Serialize data** (e.g., converting Python objects into JSON for API responses).

In FastAPI, Pydantic models are typically used to validate request bodies, query parameters, and responses. These models ensure that the input data to theAPI endpoints is in the expected format and help handle errors when the data is malformed.

### Breakdown of the `PredictionRequest` and `BatchPredictionRequest` Classes

### 1. **`PredictionRequest` Schema**

```python
class PredictionRequest(BaseModel):
    text: str

```

- **Purpose**: This schema is used for **single text predictions**.
- **Fields**:
    - `text`: A single string representing the text that you want to classify (sentiment analysis).
- **Why We Use It**:
    - **Validation**: This schema will ensure that the incoming request contains a field named `text`, and that `text` is a string. If the request doesn't conform to this format, FastAPI will automatically return a validation error.
    - **Serialization**: FastAPI will automatically convert this data into a Python object (an instance of `PredictionRequest`), making it easy to access the `text` in theendpoint logic.
- **Example Use Case**: If theendpoint is designed to receive a **single text input**, such as:

FastAPI will automatically convert the JSON data into a Python object where you can access `prediction_request.text`.
    
    ```json
    {
      "text": "I love this product!"
    }
    
    ```
    

### 2. **`BatchPredictionRequest` Schema**

```python
class BatchPredictionRequest(BaseModel):
    texts: List[str]

```

- **Purpose**: This schema is used for **batch text predictions**.
- **Fields**:
    - `texts`: A list of strings, where each string is a piece of text that you want to classify. It allows you to send multiple texts for prediction at once.
- **Why We Use It**:
    - **Validation**: The schema ensures that the incoming request contains a field named `texts` that is a list of strings. If the `texts` field is missing or not a list of strings, FastAPI will return an error.
    - **Serialization**: Just like the `PredictionRequest`, FastAPI will automatically convert the incoming JSON into a Python object (an instance of `BatchPredictionRequest`), where you can access the `texts` list easily.
- **Example Use Case**: If theendpoint is designed to receive a **batch of texts**, such as:

FastAPI will convert this into a Python object, and you can access the `texts` list directly in theendpoint.
    
    ```json
    {
      "texts": ["I love this product!", "This is the worst purchase I have ever made."]
    }
    
    ```
    

### Benefits of Using Pydantic Schemas

1. **Data Validation**:
    - Pydantic schemas automatically validate that the incoming request data matches the structure you expect. For example:
        - If a user sends an invalid request like `{ "text": 123 }` (where `123` is not a string), FastAPI will immediately respond with a validation error: `"value is not a valid string"`.
        - Similarly, if `texts` is not a list of strings (e.g., if the user sends a string instead of a list), FastAPI will raise a validation error.
2. **Error Handling**:
    - If the input data doesn’t conform to the expected structure, FastAPI will automatically return a clear error response, with a detailed message describing what was wrong with the request data (e.g., `"field required"`, `"str type expected"`, etc.).
3. **Type Safety and Autocompletion**:
    - Using Pydantic models allows you to have **type safety** in thecode. the`PredictionRequest` and `BatchPredictionRequest` classes specify exactly what type of data is expected, reducing the likelihood of runtime errors.
    - If you are using an IDE like VSCode, the **autocompletion** feature will work for fields like `prediction_request.text`, making it easier to work with the data.
4. **Automatic Serialization and Deserialization**:
    - Pydantic models handle both **serialization** (converting Python objects into JSON) and **deserialization** (converting JSON into Python objects) for you. This means you don’t have to manually parse or format the data, making thecode simpler and more robust.
5. **Clear API Documentation**:
    - FastAPI automatically generates **interactive API documentation** using the Pydantic models. When you use these schemas, FastAPI will display them in the auto-generated documentation (e.g., Swagger UI), so users and developers can see exactly what structure the input data should have.
    
    For example, the API documentation for the `PredictionRequest` schema would show:
    
    - `text`: A string, required.
    
    And for `BatchPredictionRequest`:
    
    - `texts`: A list of strings, required.

### Summary

- **Pydantic models** like `PredictionRequest` and `BatchPredictionRequest` ensure that incoming request data is validated and properly formatted before theendpoint logic processes it.
- These schemas make thecode **cleaner**, **safer**, and **more maintainable**, providing data validation, type checking, error handling, and clear API documentation automatically.
- They also allow FastAPI to handle complex operations like **serialization** and **deserialization**, saving you from writing manual code for handling JSON parsing and data conversion.

---

This modular structure ensures a clean separation of concerns, making the backend extensible and maintainable. Let me know if you'd like further clarifications or enhancements!

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/4f625cdc-6f47-45ad-92d5-2af6af956beb/e5d11e65-89b4-470c-8b81-72536b2edd97/image.png)