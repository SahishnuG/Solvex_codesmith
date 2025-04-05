import time
from quart import Quart, request, jsonify
from quart_cors import cors
import asyncio
from crewai import Crew, Agent, Task
from crewai import LLM
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from langchain.tools import BaseTool
from crewai_tools import SerperDevTool
from typing import Optional, Type, Any, Dict, List, Union

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') 
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY if GEMINI_API_KEY else ""
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# Initialize LLM
try:
    llm = LLM(model="gemini/gemini-1.5-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

# Initialize vector DB components
model = SentenceTransformer('all-MiniLM-L6-v2')
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

# Vector DB functions
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

# Custom tool classes
class LogRetrievalTool(BaseTool):
    name: str = "Log Retrieval Tool"
    description: str = "Retrieve the most recent logs from the vector database."
    
    def _run(self, query: str = "", k: int = 10) -> str:
        """
        Retrieve the most recent logs from the vector database.
        
        Args:
            query (str): Optional query parameter (not used for latest logs)
            k (int): Number of recent logs to retrieve
            
        Returns:
            str: Retrieved logs as a formatted string
        """
        global logs_metadata
        
        # Get the last k logs based on their ID (assuming higher ID = more recent)
        # Sort logs by ID in descending order and take the first k
        recent_logs = sorted(logs_metadata, key=lambda x: x["id"], reverse=True)[:k]
        
        if not recent_logs:
            return "No logs found in the database."
        
        # Format results
        formatted_logs = "Most Recent Logs:\n\n"
        for i, result in enumerate(recent_logs):
            formatted_logs += f"Log #{i+1}:\n"
            formatted_logs += f"{result['log']}\n"
            formatted_logs += f"(Timestamp: {result['timestamp']})\n\n"
        
        return formatted_logs
    
    async def _arun(self, query: str = "", k: int = 10) -> str:
        """Async implementation of log retrieval"""
        return self._run(query, k)

class EnhancedSerperTool(SerperDevTool):
    name: str = "Enhanced Search Tool"
    description: str = "Search the web for information about technologies, errors, and solutions."
    
    def _run(self, query: str) -> str:
        try:
            result = super()._run(query)
            return result
        except Exception as e:
            print(f"SerperDevTool error: {e}")
            return f"Error using search tool: {str(e)}. Please try a different approach to answer the query."

# Create INSTANCES of the tools (important!)
log_retrieval_tool = LogRetrievalTool()
enhanced_serper_tool = EnhancedSerperTool()

# --- Log Analysis Agent ---
log_analyzer = Agent(
    role="Log Analysis Bot",
    goal="Analyze logs from multiple sources, detect anomalies, and provide insights. You will retrieve relevant logs using the LogRetrievalTool.",
    backstory="A highly skilled AI specializing in log analysis, capable of detecting issues, anomalies, and trends across different logs.",
    verbose=True,
    memory=True,
    allow_delegation=False,
    tools=[log_retrieval_tool, enhanced_serper_tool],  # Pass tool instances, not classes
    llm=llm
)

# ---Chat bot Agent ---
chat_bot = Agent(
    role="Chat bot",
    goal="Assist users with log-related queries by leveraging log analysis and retrieved log data",
    backstory="A friendly chat bot that solves queries related to logs by using the analysis provided by the Log Analysis Bot and retrieving relevant logs directly.",
    verbose=True,
    memory=True,
    allow_delegation=True,
    tools=[log_retrieval_tool, enhanced_serper_tool],  # Pass tool instances, not classes
    llm=llm
)

# --- Log Analysis Task ---
log_analysis_task = Task(
    description="Analyze logs related to: '{query}'. First, retrieve recent logs using the LogRetrievalTool. Then identify any errors, warnings, or anomalies. Provide a summary of key insights.",
    expected_output="A detailed report of log anomalies, trends, and critical warnings based on the retrieved logs.",
    agent=log_analyzer,
    output_file="log_analysis_report.txt"  # Save the output for the chatbot to use
)

# --- Chat bot Task ---
chat_bot_task = Task(
    description="Answer the user's query: '{query}'. First, read the log analysis from the log_analysis_report.txt. Then use your LogRetrievalTool to get additional relevant logs if needed. Provide a comprehensive and helpful response.",
    expected_output="A helpful response that addresses the user's query based on log analysis and relevant logs.",
    agent=chat_bot,
    context=[log_analysis_task]  # This provides context from the log analysis task
)

# --- Crew Setup ---
log_analysis_crew = Crew(
    agents=[log_analyzer],
    tasks=[log_analysis_task],
    verbose=True,  # Increase verbosity for debugging
)

chat_bot_crew = Crew(
    agents=[log_analyzer, chat_bot],
    tasks=[log_analysis_task, chat_bot_task],
    verbose=True,  # Increase verbosity for debugging
)

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
        data = await request.get_json(force=True)
        query = data.get("query")
        
        if not query:
            return jsonify({"status": "error", "message": "No query provided for log retrieval"}), 400
        
        # Process logs using CrewAI with the query
        response = await asyncio.to_thread(
            log_analysis_crew.kickoff,
            inputs={"query": query}
        )
        
        # Read the analysis file for verification
        analysis_text = "Analysis not found"
        try:
            if os.path.exists("log_analysis_report.txt"):
                with open("log_analysis_report.txt", "r") as f:
                    analysis_text = f.read()
        except Exception as e:
            print(f"Error reading analysis file: {e}")
        
        # Return analysis results
        return jsonify({
            "status": "success",
            "analysis": str(response),
            "analysis_file_content": analysis_text
        })
    except Exception as e:
        print(f"Error in analyze_logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = await request.get_json(force=True)
        query = data.get("query")
        
        if not query:
            return jsonify({"status": "error", "message": "No query provided"}), 400
        
        # Use the chat_bot_crew to respond to the query
        response = await asyncio.to_thread(
            chat_bot_crew.kickoff, 
            inputs={"query": query}
        )
        
        return jsonify({
            "status": "success",
            "response": str(response)
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/add_logs', methods=['POST'])
async def add_logs():
    try:
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