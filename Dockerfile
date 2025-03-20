FROM mcr.microsoft.com/playwright:v1.51.1-noble

# Install Python 3 (default version) and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set python to point to the installed python3 binary
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Upgrade pip and install project dependencies
RUN pip install -r requirements.txt --root-user-action=ignore --break-system-packages

# Expose the port your server uses
EXPOSE 11435

# Don't try to run GUI applications in the container
ENV HEADLESS=true

# Command to run the server
CMD ["python", "agent_server.py"]