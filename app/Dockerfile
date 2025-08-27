# Use Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Copy app folder contents
COPY . /app

# Copy Models and Results folders from project root
COPY ../Models /Models
COPY ../Results /Results

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["main.py", "--server.port=8501", "--server.address=0.0.0.0"]
