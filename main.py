import os

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


app = FastAPI(title="AI Study Assistant")

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not configured",
            )
        _client = AsyncOpenAI(api_key=key)
    return _client


# --- Request / response models ---


class SummariseRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source text to summarise")


class SummariseResponse(BaseModel):
    summary: str


class NotesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source material to turn into notes")


class NotesResponse(BaseModel):
    notes: str


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str | None = Field(
        default=None,
        description="Optional passage or notes the answer should be grounded in",
    )


class QAResponse(BaseModel):
    answer: str


async def _chat(system: str, user: str) -> str:
    client = get_client()
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=502, detail="Empty response from model")
    return content.strip()


@app.post("/summarise", response_model=SummariseResponse)
async def summarise(body: SummariseRequest) -> SummariseResponse:
    """Summarise study material using the OpenAI API."""
    system = (
        "You are a study assistant. Produce a clear, concise summary of the user's text. "
        "Use short paragraphs or bullet points when it helps readability. "
        "Do not add information that is not implied by the text."
    )
    summary = await _chat(system, body.text)
    return SummariseResponse(summary=summary)


@app.post("/notes", response_model=NotesResponse)
async def notes(body: NotesRequest) -> NotesResponse:
    """Generate structured study notes from material using the OpenAI API."""
    system = (
        "You are a study assistant. Turn the user's text into structured study notes. "
        "Use headings, bullet points, and bold key terms where useful. "
        "Stay faithful to the source; do not invent facts."
    )
    notes_text = await _chat(system, body.text)
    return NotesResponse(notes=notes_text)


@app.post("/qa", response_model=QAResponse)
async def qa(body: QARequest) -> QAResponse:
    """Answer a question, optionally grounded in provided context, via the OpenAI API."""
    if body.context:
        user = f"Context:\n{body.context}\n\nQuestion:\n{body.question}"
        system = (
            "You are a study assistant. Answer the question using the context when it is relevant. "
            "If the context does not contain enough information, say what is missing and answer "
            "from general knowledge only when clearly labeled as such."
        )
    else:
        user = body.question
        system = (
            "You are a study assistant. Give a clear, accurate, and concise answer to the student's question."
        )
    answer = await _chat(system, user)
    return QAResponse(answer=answer)
