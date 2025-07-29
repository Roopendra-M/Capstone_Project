FROM python:3.10-slim

WORKDIR /app

# Copy app code
COPY flask_app/ /app/

# Copy model
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Copy pre-downloaded nltk data
COPY nltk_data /root/nltk_data

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for NLTK data path
ENV NLTK_DATA=/root/nltk_data

EXPOSE 5000

CMD ["python", "app.py"]
