# Use a specific, slim base image
FROM python:3.9-slim

# Set up a non-root user for better security
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
USER appuser

# Copy only necessary files and set ownership
# Assumes faiss_index.index and Knowledge.pdf are in your build context
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser faiss_index.index .
COPY --chown=appuser:appuser Knowledge.pdf .

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Pre-download the model (this is a great optimization)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose the port
EXPOSE 8000

# Set the command to run the application
# Note: Add workers based on your server's CPU cores, e.g., --workers 2
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]