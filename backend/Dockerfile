# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the dependencies file to the working directory
COPY ./requirements.txt /app/requirements.txt

# 4. Install any needed packages specified in requirements.txt
# We install llama-cpp-python with specific build arguments for CPU usage.
# If you were deploying to a GPU instance, you would change these args.
ARG CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV CMAKE_ARGS=${CMAKE_ARGS}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application's code
COPY ./app /app/app

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run the application
# This command runs the Uvicorn server, which is a standard for FastAPI.
# It will look for an 'app' instance in the 'main.py' file inside the 'app' module.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
