import time
from quart import Quart, request, jsonify
from quart_cors import cors
import asyncio
from crewai import Crew, Agent, Task
from dotenv import load_dotenv
import os
from crewai import LLM

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') 
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY if GEMINI_API_KEY else ""
os.environ['SERPER_API_KEY']= os.getenv('SERPER_API_KEY')
serper_tool= SerperDevTool()

# Initialize LLM
try:
    llm = LLM(model="gemini/gemini-1.5-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

# --- Log Analysis Agent ---
log_analyzer = Agent(
    role="Log Analysis Bot",
    goal="Analyze logs from multiple sources, detect anomalies, and provide insights. You are given the following logs",
    backstory="A highly skilled AI specializing in log analysis, capable of detecting issues, anomalies, and trends across different logs.",
    verbose=True,
    memory=True,
    allow_delegation=False,
    tools=[serper_tool],
    llm=llm
)
# ---Chat bot Agent ---
chat_bot = Agent(
    role="Chat bot",
    goal="Use the given logs and log analysis and solve the users queries",
    backstory="A friendly chat bot that solves all queries related to the given logs",
    verbose=True,
    memory=True,
    allow_delegation=True,
    llm=llm
)

error_finder = Agent(
    role="Error finder",
    goal="Use the given logs and try to predict future errors",
    backstory="An expert log predictive analyser",
    verbose=True,
    memory=True,
    allow_delegation=True,
    tools=[serper_tool],
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
    expected_output="Response to the given user query",
    agent=chat_bot
)

error_finder_task = Task(
    description="Analyze the provided logs and identify any potentential  errors: {logs}",
    expected_output="A list of potential errors",
    agent=error_finder
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

error_finder_crew = Crew(
    agents=[log_analyzer,error_finder],
    tasks=[log_analysis_task, error_finder_task],
    process="sequential"
)


model = SentenceTransformer('all-MiniLM-L6-v2')
# Create or load FAISS index
INDEX_DIMENSION = 384  # Dimension of the embeddings from the model
INDEX_FILE = 'logs_index.faiss'
META_FILE = 'logs_metadata.json'

# Initialize index and metadata
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, 'r') as f:
        logs_metadata = json.load(f)
else:
    index = faiss.IndexFlatL2(INDEX_DIMENSION)
    logs_metadata = []

def add_logs_to_vector_db(logs, metadata=None):
    """
    Add logs to the vector database.
    
    Args:
        logs (str or list): Log text or list of log texts to be added
        metadata (dict or list, optional): Additional metadata for each log
    
    Returns:
        int: Number of logs added
    """
    global logs_metadata, index
    
    # Ensure logs is a list
    if isinstance(logs, str):
        logs = [logs]
        if metadata and not isinstance(metadata, list):
            metadata = [metadata]
    
    # Create embeddings
    embeddings = model.encode(logs)
    
    # Add to FAISS index
    index.add(np.array(embeddings).astype('float32'))
    
    # Store metadata
    for i, log in enumerate(logs):
        log_entry = {
            "id": len(logs_metadata),
            "log": log,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add custom metadata if provided
        if metadata and i < len(metadata):
            log_entry.update(metadata[i])
            
        logs_metadata.append(log_entry)
    
    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, 'w') as f:
        json.dump(logs_metadata, f)
    
    return len(logs)

def search_logs_in_vector_db(query, k=5):
    """
    Search for logs in the vector database.
    
    Args:
        query (str): Query text to search for
        k (int): Number of results to return
    
    Returns:
        list: List of dictionaries containing matching logs and their metadata
    """
    # Create query embedding
    query_embedding = model.encode([query])
    
    # Search in FAISS index
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # Get results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(logs_metadata):  # Valid index
            result = logs_metadata[idx].copy()
            result['distance'] = float(distances[0][i])  # Add similarity score
            results.append(result)
    
    return results
# Create Quart app with CORS support
app = Quart(__name__)
app = cors(app)  # Enable CORS for all routes

@app.route('/')
async def home():
    return "Welcome to Log Analysis Bot!"

@app.route('/search_logs', methods=['GET', 'POST'])
async def search_logs():
    try:
        if request.method == 'POST':
            data = await request.get_json(force=True)
            query = data.get("query", "")
            limit = data.get("limit", 5)
        else:
            query = request.args.get("query", "")
            limit = int(request.args.get("limit", 5))
        
        if not query:
            return jsonify({"status": "error", "message": "No query provided"}), 400
        
        results = search_logs_in_vector_db(query, limit)
        
        return jsonify({
            "status": "success",
            "count": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analyze_logs', methods=['POST'])
async def analyze_logs():
    try:
        # Use force=True to handle cases where Content-Type header is not set correctly
        data = await request.get_json(force=True)
        query = data.get("query")
        limit = data.get("limit", 5)  # Default to retrieving 5 logs
        
        if not query:
            return jsonify({"status": "error", "message": "No query provided for log retrieval"}), 400
        
        # Directly use the search_logs_in_vector_db function
        log_results = search_logs_in_vector_db(query, limit)
        
        if not log_results:
            return jsonify({"status": "error", "message": "No matching logs found in database"}), 404
        
        # Extract the actual log content from results
        logs = [result["log"] for result in log_results]
        logs_text = "\n".join(logs)
        
        # Process logs using CrewAI
        response = await asyncio.to_thread(log_analysis_crew.kickoff,
                                          inputs={"logs": logs_text})
        future_errors = await asyncio.to_thread(chat_bot_crew.kickoff,
                                          inputs={"query": "Give a list of potential future errors using the given logs"})
        
        # Return analysis results along with metadata about retrieved logs
        return jsonify({
            "status": "success",
            "analysis": str(response),
            "logs_analyzed": len(logs),
            "errors": str(future_errors),
            "search_results": log_results
        })
    except Exception as e:
        print(f"Error in analyze_logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/chat/<uinput>', methods=['POST'])
async def chat(uinput):
    try:
        data = await request.get_json(force=True)
        query = data.get("query")
        
        if not query:
            return jsonify({"status": "error", "message": "No query provided"}), 400
        
        # First, get relevant logs
        log_results = search_logs_in_vector_db(query, k=10)
        logs_text = "\n".join([result["log"] for result in log_results])
        
        # Use the chat_bot_crew to respond to the query
        response = await asyncio.to_thread(
            chat_bot_crew.kickoff, 
            inputs={
                "logs": str(logs_text),
                "query": str(uinput)
            }
        )
        return jsonify({
            "status": "success",
            "response": str(response),
            "related_logs_count": len(log_results)
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/add_logs', methods=['POST'])
async def add_logs():
    try:
        # Use force=True to handle cases where Content-Type header is not set correctly
        data = await request.get_json(force=True)
        logs = data.get("logs", "")
        metadata = data.get("metadata", None)
        
        # Convert logs to JSON if received as a string and it appears to be JSON
        if isinstance(logs, str) and logs.strip().startswith(("{", "[")):
            try:
                logs_data = json.loads(logs)
                # If it's parsed to a dict/object, store it as a JSON string
                if isinstance(logs_data, dict):
                    logs = json.dumps(logs_data)
            except json.JSONDecodeError:
                # If it's not valid JSON, keep it as a string
                pass
        
        if not logs:
            return jsonify({"status": "error", "message": "No logs provided"}), 400
        
        count = add_logs_to_vector_db(logs, metadata)
        
        return jsonify({
            "status": "success", 
            "message": f"Added {count} logs to vector database"
        })
    except Exception as e:
        print(f"Error in add_logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)