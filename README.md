# AI Study Assistant

## Overview

This project is a lightweight AI-powered study assistant built with FastAPI and the OpenAI API.  
It provides endpoints for text summarisation, structured note generation, and Q&A, making it easier to work with study materials.

## Features

- `/summarise`: Produce a clear, concise summary of input text
- `/notes`: Turn raw text into structured study notes with headings and bullet points
- `/qa`: Answer questions, optionally grounded in a given context passage

## Tech Stack

- Backend: FastAPI (Python)
- AI: OpenAI `gpt-4o-mini` via the async client (`AsyncOpenAI`)
- Data modelling: Pydantic request/response models

## How It Works

1. The client sends a JSON request to one of the endpoints.
2. The API constructs a system prompt and user message based on the endpoint.
3. The service calls the OpenAI Chat Completions API through a shared helper function.
4. The response is validated and returned as a structured JSON payload.

The OpenAI client is lazily initialised using the `OPENAI_API_KEY` environment variable.  
If the key is missing, the API responds with a 503 error so configuration issues surface early.

## Endpoints

- `POST /summarise`
- `POST /notes`
- `POST /qa`

You can explore and test them via the built-in Swagger UI at `/docs` once the server is running.

## Possible Future Improvements

- Add authentication and basic rate limiting
- Add a simple frontend (React or Next.js) to interact with the assistant
- Introduce retrieval-augmented generation (RAG) using a vector store
- Log prompts and responses for analysis and prompt optimisation

## Author

Jingjing Wang
