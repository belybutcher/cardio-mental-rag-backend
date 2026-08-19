from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import get_rag_chain

app = FastAPI(title="Cardio & Mental Health RAG API")

# Initialize the chain on startup
chain = get_rag_chain()

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        response_text = chain.invoke(request.question)
        return ChatResponse(answer=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
