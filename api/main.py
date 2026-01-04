from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Credit Risk Engine API")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Credit Risk Engine API"} 