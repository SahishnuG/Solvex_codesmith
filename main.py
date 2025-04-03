import time
from quart import Quart, request, jsonify
from quart_cors import cors
import asyncio
from crewai import Crew, Agent, Task
from dotenv import load_dotenv
import os
from crewai import LLM

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') 
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY if GEMINI_API_KEY else ""

# Initialize LLM
try:
    llm = LLM(model="gemini/gemini-1.5-flash")
except Exception as e:
    llm = None

# --- Log Analysis Agent ---
log_analyzer = Agent(
    role="Log Analysis Bot",
    goal="Analyze logs from multiple sources, detect anomalies, and provide insights. You are given the following logs: {logs}",
    backstory="A highly skilled AI specializing in log analysis, capable of detecting issues, anomalies, and trends across different logs.",
    verbose=False,
    memory=True,
    allow_delegation=False,
    llm=llm
)

chat_bot = Agent(
    role="Chat bot",
    goal="Use the given logs and log analysis and solve the users queries",
    backstory="A friendly chat bot that solves all queries related to the given logs",
    verbose=False,
    memory=True,
    allow_delegation=True,
    llm=llm
)

# --- Log Analysis Task ---
log_analysis_task = Task(
    description="Analyze the provided logs and identify any errors, warnings, or anomalies: {logs}\n Provide a summary of key insights.",
    expected_output="A detailed report of log anomalies, trends, and critical warnings.",
    agent=log_analyzer
)
# --- Chat bot Task ---
chat_bot_task = Task(
    description="{query}",
    expected_output="Respose to the given user query",
    agent=chat_bot
)

# --- Crew Setup ---
log_analysis_crew = Crew(
    agents=[log_analyzer],
    tasks=[log_analysis_task]
)

chat_bot_crew = Crew(
    agents=[log_analyzer, chat_bot],
    tasks=[log_analysis_task, chat_bot_task]
)

# Create Quart app with CORS support
app = Quart(__name__)
app = cors(app)  # Enable CORS for all routes

@app.route('/')
async def home():
    return "Welcome to Log Analysis Bot!"

@app.route('/analyze_logs', methods=['POST'])  
async def analyze_logs():
    try:
        data = await request.json
        logs = data.get("logs", "")
        if not logs:
            return jsonify({"error": "No logs provided"}), 400
        
        # Process logs using CrewAI
        response = await asyncio.to_thread(log_analysis_crew.kickoff, 
                                           inputs={"logs": logs})
        
        return jsonify({"analysis": str(response)})  # Convert to string
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/chat/<query>', methods=['POST'])  
async def chat(query):
    try:
        data = await request.json
        logs = data.get("logs", "")
        if not logs:
            return jsonify({"error": "No logs provided"}), 400
        
        # Process logs using CrewAI
        response = await asyncio.to_thread(chat_bot_crew.kickoff, 
                                           inputs={"logs": logs,"query": query})
        
        return jsonify({"response": str(response)})  # Convert to string
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
