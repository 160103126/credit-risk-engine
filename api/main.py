from fastapi import FastAPI
from .routes import router

from prometheus_client import make_asgi_app

app = FastAPI(title="Credit Risk Engine API")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Credit Risk Engine API"} 