# FastAPI Document Retrieval Application

## Overview

This project is a FastAPI-based document retrieval application that leverages Redis for caching and data storage. It utilizes the `SentenceTransformer` model to provide semantic search capabilities, enabling efficient and accurate document retrieval. The application is containerized using Docker to ensure a consistent development and deployment environment.

## Features

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **Redis**: An in-memory data store used for caching, data storage, and message brokering. Redis is utilized to cache search results, store documents, and manage user data, ensuring fast access and reduced latency.
- **Sentence-Transformers**: Provides semantic search capabilities using pre-trained models to deliver contextually relevant search results.
- **Docker**: Containerizes the application to ensure consistency across different environments and streamline deployment.

## Getting Started

### Prerequisites

- **Docker**: Make sure Docker and Docker Compose are installed on your machine.
- **Python**: Required for local development and testing.

### Project Structure

- **`app.py`**: Contains the main FastAPI application code, including endpoints for health checks and search operations. It also includes logic for interacting with Redis and performing semantic searches.
- **`Dockerfile`**: Defines the steps to build the Docker image for the FastAPI application. It sets up the Python environment, installs dependencies, and configures the application to run within a Docker container.
- **`docker-compose.yml`**: Defines the services and configuration for Docker Compose. It sets up both the FastAPI application and Redis containers, ensuring they run on the same Docker network and can communicate with each other.
