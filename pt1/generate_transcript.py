from dotenv import load_dotenv
from openai import OpenAI
import os, uuid, random, json, datetime as dt

load_dotenv()
client = OpenAI()

def create_transcript(prompt):
    """Use OpenAI API to create a dummy transcript and returns the transcript formatted as a JSON string"""
    # I used Salesloft docs and GPT to create this schema mirroring the “transcription” endpoint from SL
    TRANSCRIPTION_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id", "language_code", "created_at", "updated_at",
            "conversation", "transcription_sentences"
        ],
        "properties": {
            "id": {"type": "string", "description": "UUID for the transcription"},
            "language_code": {"type": "string", "pattern": "^[a-z]{2}-[A-Z]{2}$"},
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
            "conversation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "_href"],
                "properties": {
                    "id": {"type": "string"},
                    "_href": {"type": "string"}
                }
            },
            "transcription_sentences": {
                "type": "array",
                "minItems": 10,
                "maxItems": 80,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "speaker", "timestamp_start", "timestamp_end", "text"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 0},
                        "speaker": {"type": "string", "enum": ["sales_rep", "prospect"]},
                        "timestamp_start": {"type": "string", "pattern": "^\\d{2}:\\d{2}:\\d{2}$"},
                        "timestamp_end": {"type": "string", "pattern": "^\\d{2}:\\d{2}:\\d{2}$"},
                        "text": {"type": "string", "minLength": 1, "maxLength": 1000}
                    }
                }
            }
        }
    }
    response = client.responses.create(
        model="gpt-4.1-2025-04-14", #Using GPT4.1 so I can pass in temperature paramater and add another layer of variation 
        temperature=0.9,
        text={"format": {"type": "json_object"}},  # Per "Structured Output" documentation
        input=[
            {"role": "system", "content": "You are a realistic sales call transcript generator."},
            {"role": "user", "content": prompt}, # Pass in prompt that GPT made above
        ],
    )
    json_str = response.output_text   
    return(json_str)

    
def build_prompt():
    # On my first couple runs I was running into issues where the transcripts were too similar so I asked GPT to help generate some parameters I could pass into the prompt. I tuned these parameters based on my own experience and reviewed the code to ensure I understood the f-string logic
    
    outcomes = [
        "discovery_meeting_scheduled",
        "followup_email_requested",
        "hard_no",
        "callback_requested",
    ] # Most common outcomes from connected calls in my experience
    
    # Scenario variations for call
    industries = ["fintech", "healthcare", "gaming", "retail", "SaaS", "adtech"]
    personas = ["CTO", "Head of Data", "VP Engineering", "Director of IT", "CFO", "Product Manager"]
    current_stacks = ["PostgreSQL", "MySQL", "DynamoDB", "Snowflake", "Firebase", "on-prem MongoDB"]
    triggers = [
        "recent migration initiative",
        "cost optimization push",
        "scaling challenges",
        "data compliance review",
        "AI feature rollout",
        "microservices modernization",
    ]
    tones = ["consultative", "direct", "curious", "friendly", "concise"]
    tech_objections = ["performance under load", "multi-region failover", "migration risk", "data modeling effort"]
    biz_objections  = ["pricing", "vendor lock-in", "security/compliance", "contract terms"]

    # Randomize the scenario
    industry = random.choice(industries)
    persona = random.choice(personas)
    stack = random.choice(current_stacks)
    trigger = random.choice(triggers)
    tone = random.choice(tones)
    tech_objection = random.choice(tech_objections)
    biz_objection = random.choice(biz_objections)
    outcome = random.choice(outcomes)

    # Use f-string to format prompt with the above variations
    return f"""
    Generate a realistic cold-call transcription where the sales rep is pitching MongoDB Atlas.

    Scenario context:
    - Industry: {industry}
    - Prospect: {persona}
    - Current database stack: {stack}
    - Recent trigger: {trigger}
    - Desired outcome: {outcome}
    - Potential topics of objection: {biz_objection} and/or {tech_objection}
    - Conversation tone: {tone}

    Requirements:
    - The Rep name should always be Arjun and the prospect should always be named Zhi. No last names
    - The prospect's company name should always be Acme Corp.
    - Natural dialogue with objections (price, migration risk, vendor lock-in, performance, security/compliance, etc.)
    - 1–3 minutes of conversation (10–80 sentences).
    - Use speaker labels "sales_rep" and "prospect".
    - Timestamps must be HH:MM:SS and strictly ascending, starting near 00:00:02.
    - Return **only** valid JSON with fields:
        id, language_code, created_at, updated_at,
        conversation {{id,_href}},
        transcription_sentences [{{index,speaker,timestamp_start,timestamp_end,text}}].
    - Make it sound human — include filler words, interruptions, and realistic tone.
    - The language_code should be "en-US".
    - Do not include any explanatory text outside the JSON.
    """


def write_json(json_string: str, filename: str):
    """Function takes a JSON string and stores it locally as Json file with the specified name in the data folder"""
    python_object = json.loads(json_string)
    # Write the Python dictionary to a JSON file
    with open(filename, "w") as file:
        json.dump(python_object, file, indent=4)
    print(f"JSON data successfully saved to '{filename}'")

def main():
    """
    Using the create_transcript and write_json functions this script uses OpenAI's API to create 25 mock sales call transcripts 
    """
    for i in range(1, 26):
        outputfile = f"data/transcript{i}.json"
        
        # Basic error handling in case of faillures during API calls or funcition calls (used GPT for initial guidance on formatting)
        try:
            prompt = build_prompt()
        except Exception as e:
            print(f"Prompt build failled for Transcript {i}")
            sys.exit(1)
        try:
            transcript = create_transcript(prompt) 
        except Exception as e: 
            print(f"Transcript {i} failed to create")
            sys.exit(1)
        try:
            write_json(transcript, outputfile)
        except Exception as e: 
            print(f"Transcript {i} failed to be saved as a JSON")
            sys.exit(1)
    print("Transcripts created!")

if __name__ == "__main__":
    main()