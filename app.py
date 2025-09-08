import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models (load once at startup)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

class FailureRequest(BaseModel):
    test_name: str
    error_message: str
    stack_trace: str
    history: list = []  # Previous failures for context

class ClusterRequest(BaseModel):
    failures: list  # List of FailureRequest objects

@app.post("/analyze-failure")
async def analyze_failure(request: FailureRequest):
    try:
        # Generate embedding for deduplication
        error_text = f"{request.test_name}: {request.error_message}"
        embedding = sentence_model.encode([error_text])[0].tolist()
        
        # AI analysis for severity and root cause
        analysis = await generate_analysis(request)
        
        return {
            "embedding": embedding,
            "severity": analysis["severity"],
            "probable_cause": analysis["probable_cause"],
            "summary": analysis["summary"],
            "is_duplicate": False  # Will be determined after comparing with history
        }
    except Exception as e:
        logger.error(f"Error analyzing failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-duplicates")
async def check_duplicates(failures: list, new_embedding: list):
    try:
        if not failures:
            return {"is_duplicate": False, "similar_ticket": None}
        
        # Compare with existing failures
        existing_embeddings = [f["embedding"] for f in failures]
        similarities = sentence_model.similarity([new_embedding], existing_embeddings)[0]
        
        max_similarity = max(similarities) if similarities.size > 0 else 0
        most_similar_index = np.argmax(similarities) if similarities.size > 0 else -1
        
        is_duplicate = max_similarity > 0.85  # Threshold for duplicates
        
        return {
            "is_duplicate": is_duplicate,
            "similarity_score": float(max_similarity),
            "similar_ticket": failures[most_similar_index] if most_similar_index != -1 else None
        }
    except Exception as e:
        logger.error(f"Error checking duplicates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster-failures")
async def cluster_failures(request: ClusterRequest):
    try:
        if len(request.failures) < 2:
            return {"clusters": []}
        
        # Generate embeddings for all failures
        texts = [f"{f['test_name']}: {f['error_message']}" for f in request.failures]
        embeddings = sentence_model.encode(texts)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
        labels = clustering.labels_
        
        # Group failures by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(request.failures[i])
        
        return {"clusters": clusters}
    except Exception as e:
        logger.error(f"Error clustering failures: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_analysis(request):
    """Use OpenAI API to generate analysis"""
    prompt = f"""
    Analyze this test failure and provide:
    1. Severity level (Critical, Major, Minor)
    2. Probable root cause
    3. A concise summary for a Jira ticket
    
    Test: {request.test_name}
    Error: {request.error_message}
    Stack trace: {request.stack_trace[:1000]}  # Truncate if too long
    
    Respond in JSON format:
    {{
        "severity": "Major",
        "probable_cause": "Brief explanation",
        "summary": "Concise ticket summary"
    }}
    """
    
    try:
        # Uncomment when you have OpenAI API key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        analysis = json.loads(response.choices[0].message.content)
        
        # For now, return mock response
        return {
            "severity": "Major",
            "probable_cause": "Potential null pointer exception in user creation flow",
            "summary": f"Test Failure: {request.test_name} - {request.error_message[:100]}..."
        }
    except Exception as e:
        logger.warning(f"OpenAI API failed, using fallback: {str(e)}")
        return {
            "severity": "Minor",
            "probable_cause": "Unable to analyze with AI",
            "summary": f"Test Failure: {request.test_name}"
        }

