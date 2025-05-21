# Use official Python image
FROM python:3.10.17-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
