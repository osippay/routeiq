FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create state directory
RUN mkdir -p state

EXPOSE 8000

CMD ["python", "-m", "app.server"]
