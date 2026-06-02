# Gubat Server

**Gubat Server** is a high-performance, secure SMS Spam Filter API built with FastAPI. It leverages a combination of TF-IDF vectorization, sentiment analysis (VADER), and extensive handcrafted feature extraction to accurately classify SMS messages as spam or ham. The core classification model is powered by Yggdrasil Decision Forests (YDF). To be used with [Gubat Messages](https://github.com/doidu/gubat-messages).

## 🌟 Features

- **Advanced Text Preprocessing**: Custom standardization including lowercase conversion, leetspeak decoding (e.g., `m3ssage` → `message`), and removal of English and Tagalog stopwords.
- **Comprehensive Feature Extraction**: Analyzes message length, digit/space/punctuation ratios, and detects patterns like DTI codes, reference numbers, URLs, phone numbers, emails, and spam/urgency/money keywords.
- **Sentiment Analysis**: Integrates NLTK's VADER to extract emotional intensity and polarity features.
- **Secure by Design**: Incoming messages are decrypted using AES-256 GCM before processing.
- **API Authentication**: Secured endpoints using Bearer token (API Key) authentication.
- **Containerized**: Ready for deployment with included `Dockerfile`.

## 🛠️ Tech Stack

- **Framework**: FastAPI, Uvicorn
- **Machine Learning**: Yggdrasil Decision Forests (`ydf`), TensorFlow (TextVectorization), Scikit-Learn (StandardScaler)
- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK (VADER Sentiment Analyzer)
- **Security**: Cryptography (AES-256 GCM)

## 📋 Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerized deployment)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/doidu/gubat-server.git
cd gubat-server
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: The server will automatically download the `vader_lexicon` for NLTK on first run if it's not already present.*

### 4. Configure Environment Variables
Create a `.env` file in the root directory or export the following environment variables:
```bash
export API_KEY="your_secure_api_key_here"
export ENCRYPTION_KEY="your_32_byte_encryption_key_here"
```
> **Important**: The `ENCRYPTION_KEY` must be a string that can be encoded to at least 32 bytes for AES-256 encryption.

## 🏃 Running the Server

### Local Development
```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
```bash
# Build the image
docker build -t gubat-server .

# Run the container
docker run -d -p 8000:8000 \
  -e API_KEY="your_secure_api_key_here" \
  -e ENCRYPTION_KEY="your_32_byte_encryption_key_here" \
  --name gubat-server-app \
  gubat-server
```

## 📡 API Endpoints

All endpoints require a valid API key passed in the `Authorization` header as a Bearer token.

### Health Check
- **Endpoint**: `GET /health`
- **Description**: Verifies that the API is running and healthy.
- **Headers**: `Authorization: Bearer <your_api_key>`
- **Response**:
  ```json
  {
    "status": "healthy",
    "message": "GubatFilterAPI is running"
  }
  ```

### Predict Spam
- **Endpoint**: `POST /predict`
- **Description**: Accepts a list of base64-encoded, AES-256 GCM encrypted SMS messages, decrypts them, and returns spam predictions.
- **Headers**: `Authorization: Bearer <your_api_key>`
- **Request Body**:
  ```json
  {
    "messages": [
      "base64_encoded_encrypted_message_1",
      "base64_encoded_encrypted_message_2"
    ]
  }
  ```
- **Response**:
  ```json
  {
    "predictions": ["0", "1"]
  }
  ```
  *(Note: `0` indicates Ham, `1` indicates Spam)*

## 📁 Project Structure

```text
gubat-server/
├── server.py                     # Main FastAPI application and feature extraction logic
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration for containerization
├── gubat-docu.pdf                # Detailed project documentation
├── gubat_model/                  # Pre-trained YDF model files
├── gubat_model_configs/          # Model configuration files (scaler, TF-IDF vocab/weights, feature names, stopwords)
└── gubat_model_training_and_results/ # Jupyter notebooks, tuning results, and dataset used for training
```

## 📜 License

This project is proprietary. Please refer to the repository owner for licensing details.

## 📞 Contact

For questions or support, please open an issue in the repository or contact the maintainer.
