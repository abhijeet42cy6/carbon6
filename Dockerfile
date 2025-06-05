# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8001
EXPOSE 8001

# Command to run FastAPI app
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8001"]

# NVIDIA api key
ENV NVIDIA_API_KEY=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
