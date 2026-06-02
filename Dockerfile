# Start from a stable Python 3.12 image
FROM python:3.12.12-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (like build-essentials, for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to cache this layer
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK stopwords data
RUN python -m nltk.downloader vader_lexicon

# Copy all your project files into the container's /app directory
COPY . .

# Tell Docker that the container will listen on port 8000
EXPOSE 8000

# The command to run when the container starts
# This runs your server on 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
