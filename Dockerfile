FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Render assigns the port via the PORT environment variable
EXPOSE 10000

CMD gunicorn app:app --bind 0.0.0.0:$PORT
