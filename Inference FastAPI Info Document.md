# **FastAPI Inference Service Documentation**

This document provides comprehensive information on the FastAPI inference service. It covers setup, endpoints, request-response formats, and troubleshooting tips for seamless interaction.

---

## **Overview**

The FastAPI application provides endpoints for performing single or batch predictions using a machine learning model. The service is designed for high performance, scalability, and ease of integration with other systems.

---

## **Key Features**
1. **Health Check Endpoint**: Verify the service's availability.
2. **Single Prediction**: Perform predictions on individual inputs.
3. **Batch Prediction**: Process multiple inputs in a single request for efficiency.
4. **Fast and Scalable**: Built using FastAPI for asynchronous handling of multiple requests.
5. **Easy Deployment**: Compatible with Docker, cloud services, and local systems.
6. **Model and Tokenizer Integration**: Automatically loads the pre-trained model and tokenizer for inference.

---

## **Getting Started**

### **Requirements**
1. **Python**: Version 3.8 or higher.
2. **Dependencies**: Install required packages using:
   ```bash
   pip install fastapi uvicorn transformers
   ```
3. **Running the Service**: Start the FastAPI server using:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## **Endpoints**

### 1. **Health Check**
   - **Purpose**: To verify that the service is running.
   - **Endpoint**: `/health`
   - **Method**: `GET`
   - **Request**: No input required.
   - **Response**:
     ```json
     {
       "status": "ok"
     }
     ```

---

### 2. **Single Prediction**
   - **Purpose**: Get predictions for one or more text inputs.
   - **Endpoint**: `/predict`
   - **Method**: `POST`
   - **Headers**:
     - `Content-Type: application/json`
   - **Request Body**:
     ```json
     {
       "texts": ["Your first text input", "Your second text input"]
     }
     ```
   - **Response**:
     ```json
     {
       "predictions": [1, 0]
     }
     ```
     - **`predictions`**: List of integers representing model outputs (e.g., 1 for positive, 0 for negative).

---

### 3. **Batch Prediction**
   - **Purpose**: Perform predictions on batches of text inputs.
   - **Endpoint**: `/predict_batch`
   - **Method**: `POST`
   - **Headers**:
     - `Content-Type: application/json`
   - **Request Body**:
     ```json
     {
       "batch_texts": [
         ["Input1 for batch1", "Input2 for batch1"],
         ["Input1 for batch2", "Input2 for batch2"]
       ]
     }
     ```
   - **Response**:
     ```json
     {
       "predictions": [
         [1, 1],
         [0, 0]
       ]
     }
     ```
     - **`predictions`**: Nested list of integers for each batch's predictions.

---

## **Using Postman to Test Endpoints**

1. **Health Check**:
   - **URL**: `http://localhost:8000/health`
   - **Method**: `GET`
   - **Expected Response**:
     ```json
     {
       "status": "ok"
     }
     ```

2. **Single Prediction**:
   - **URL**: `http://localhost:8000/predict`
   - **Method**: `POST`
   - **Headers**:
     - `Content-Type`: `application/json`
   - **Body**:
     ```json
     {
       "texts": ["Sample text 1", "Sample text 2"]
     }
     ```

3. **Batch Prediction**:
   - **URL**: `http://localhost:8000/predict_batch`
   - **Method**: `POST`
   - **Headers**:
     - `Content-Type`: `application/json`
   - **Body**:
     ```json
     {
       "batch_texts": [
         ["Batch1 text1", "Batch1 text2"],
         ["Batch2 text1", "Batch2 text2"]
       ]
     }
     ```

---

## **Deployment Options**

### **Local Deployment**
1. Clone the repository or place the `app.py` file in your project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the service:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

### **Docker Deployment**
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.8-slim
   WORKDIR /app
   COPY . /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Build and Run:
   ```bash
   docker build -t fastapi-app .
   docker run -p 8000:8000 fastapi-app
   ```

### **Cloud Deployment**
- Use services like AWS, Azure, or GCP to deploy the app. For example, use AWS ECS or GCP Cloud Run for containerized deployments.

---

## **Troubleshooting**

1. **Error: `Explicit proxy to non-unicast IP address forbidden`**:
   - Use `http://localhost:8000` instead of `http://0.0.0.0:8000`.

2. **CORS Issues**:
   - Add CORS middleware to the app if accessing from a browser:
     ```python
     from fastapi.middleware.cors import CORSMiddleware

     app.add_middleware(
         CORSMiddleware,
         allow_origins=["*"],
         allow_credentials=True,
         allow_methods=["*"],
         allow_headers=["*"],
     )
     ```

3. **Performance Issues**:
   - Optimize `batch_texts` size for `/predict_batch` based on your system's memory.
   - Use a GPU if available for faster inference.

---

## **Future Enhancements**
1. Add model versioning support.
2. Enable authentication and API rate limiting.
3. Integrate with monitoring tools like Prometheus for better observability.
4. Deploy as a scalable microservice using Kubernetes.

---

This document ensures you can efficiently set up, use, and troubleshoot the FastAPI inference service. Happy predicting! 🚀