# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import predict, llm_predict  # keep as-is

app = FastAPI(
    title="End-to-End ML + LLM API",
    description="API for churn prediction and LLM explanations",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(predict.router, prefix="/api")
app.include_router(llm_predict.router, prefix="/api/llm")

@app.get("/")
async def root():
    return {"message": "Welcome to the End-to-End ML + LLM API"}

@app.on_event("startup")
async def startup_event():
    print("Starting up the API...")
    
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the API...")
