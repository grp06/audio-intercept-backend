from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import os
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Audio Intercept API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TranscriptRequest(BaseModel):
    transcript: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "audio-intercept-api"
    }

@app.post("/generate-overview")
async def generate_overview(request: TranscriptRequest):
    try:
        completion = client.chat.completions.create(
            model="chatgpt-4o-latest",
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system",
                    "content": """You are the world's top military strategist analyzing Russian radio communications.
                    Analyze the provided transcript and return a JSON object with the following structure:
                    {
                        "threat_level": string (LOW, MEDIUM, HIGH, CRITICAL),
                        "sentiment": string (describing emotional tone and morale),
                        "tags": array of strings (relevant military keywords),
                        "key_insights": array of strings (important tactical/strategic observations)
                    }"""
                },
                {
                    "role": "user",
                    "content": request.transcript
                }
            ]
        )

        print(completion.choices[0].message.content)
        
        return completion.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-detailed-analysis")
async def generate_detailed_analysis(request: TranscriptRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using the latest GPT-4 model
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system",
                    "content": """You are the world's top military strategist analyzing Russian radio communications.
                    Provide a detailed analysis of the transcript and return a JSON object with the following structure:
                    {
                        "threat_level": string (LOW, MEDIUM, HIGH, CRITICAL),
                        "threat_sources": array of strings,
                        "potential_actions": array of strings,
                        "sentiment": string,
                        "morale_indicators": array of strings,
                        "psychological_state": string,
                        "operational_insights": {
                            "troop_movements": array of strings,
                            "logistics": array of strings,
                            "command_structure": array of strings
                        },
                        "tactical_insights": array of strings,
                        "strategic_insights": array of strings,
                        "weaknesses": array of strings,
                        "communication_patterns": {
                            "frequency": string,
                            "urgency": string,
                            "code_words": array of strings
                        },
                        "key_entities": array of strings,
                        "relationships": object (key-value pairs of relationships),
                        "geospatial_info": {
                            "locations": array of strings,
                            "movement_patterns": array of strings
                        },
                        "risk_assessment": {
                            "immediate_risks": array of strings,
                            "potential_scenarios": array of strings
                        },
                        "recommendations": {
                            "countermeasures": array of strings,
                            "priority_actions": array of strings
                        }
                    }
                    
                    Analyze the transcript thoroughly and ensure all fields are populated with relevant information.
                    If certain information is not available in the transcript, provide reasoned assumptions based on 
                    the available context. Focus on actionable intelligence and strategic insights."""
                },
                {
                    "role": "user",
                    "content": request.transcript
                }
            ]
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
