import os 
import json
import uuid
import base64
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, join_room, emit
import av  # PyAV for conversion
import azure.cognitiveservices.speech as speechsdk

# --------------------- Agentic Imports --------------------- #
from typing import TypedDict, Annotated, List, Optional
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from openai import OpenAI
from langchain_core.tools import tool

# --------------------- Embedding & FAISS Imports --------------------- #
from sentence_transformers import SentenceTransformer
import faiss

# --------------------- Flask App Setup --------------------- #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global conversation log file (persisted on server)
LOG_FILE = 'conversation_logs.json'
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        try:
            conversation_logs = json.load(f)
        except json.JSONDecodeError:
            conversation_logs = []
else:
    conversation_logs = []

# Global active ticket tracking and mode storage
active_tickets = set()
active_modes = {}  # ticket_id -> mode ("audio" or "text")

# Global QA data storage: mapping ticket_id -> { index, texts, file_names }
qa_data = {}

# --------------------- Azure Speech Configuration --------------------- #
speech_key = "EEWU2xnaw1HcrsnsOFPhRF1EG50Ef0Gn0hFgxbHnm8E7EpTd3Uc6JQQJ99BCACYeBjFXJ3w3AAAAACOGNbAC"
service_region = "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["te-IN", "hi-IN", "de-DE"])

# --------------------- SentenceTransformer and FAISS Setup --------------------- #
embedding_model = SentenceTransformer('mxbai-embed-large-v1')

def get_embedding(text: str) -> np.ndarray:
    embedding = embedding_model.encode(text)
    return embedding

def create_embeddings_from_folder(folder_path: str):
    embeddings = []
    texts = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                embedding = get_embedding(text)
                embeddings.append(embedding)
                texts.append(text)
                file_names.append(filename)
    return embeddings, texts, file_names

def create_faiss_index(embeddings):
    d = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(d)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

# --------------------- OpenAI LLM Setup --------------------- #
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key="###"
)

def get_llm_response(prompt: str) -> str:
    completion = client.chat.completions.create(
      model="nvidia/llama-3.1-nemotron-70b-instruct",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.5,
      top_p=1,
      max_tokens=1024,
      stream=True
    )
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    return response_text

# --------------------- Flask Routes --------------------- #
@app.route('/', methods=['GET', 'POST'])
@app.route('/client-login', methods=['GET', 'POST'])
def client_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            df = pd.read_excel("clients.xlsx")
        except Exception as e:
            return f"Error reading clients.xlsx: {e}"
        valid = ((df['username'] == username) & (df['password'] == password)).any()
        if valid:
            ticket_id = str(uuid.uuid4())
            session['ticket_id'] = ticket_id
            session['username'] = username
            active_tickets.add(ticket_id)
            return redirect(url_for('client_mode'))
        else:
            return render_template("client_login.html", error="Invalid credentials. Please try again.")
    return render_template("client_login.html")

@app.route('/client-mode', methods=['GET', 'POST'])
def client_mode():
    if 'ticket_id' not in session:
        return redirect(url_for('client_login'))
    if request.method == 'POST':
        mode = request.form.get('mode')
        session['mode'] = mode
        return redirect(url_for('client'))
    return render_template("client_mode.html", ticket_id=session['ticket_id'], username=session.get('username'))

@app.route('/client')
def client():
    if 'ticket_id' not in session or 'mode' not in session:
        return redirect(url_for('client_login'))
    return render_template("client.html", ticket_id=session['ticket_id'], mode=session['mode'])

@app.route('/agent')
def agent():
    return render_template("agent.html")

@app.route('/tickets')
def tickets():
    return render_template("tickets.html", tickets=active_tickets)

@app.route('/logs')
def logs():
    return jsonify(conversation_logs)

# New route for Followup Alerts page
@app.route('/followup_alerts')
def followup_alerts():
    alerts = [entry for entry in conversation_logs if entry.get("role") == "followup_agent"]
    return render_template("followup_alerts.html", alerts=alerts)

# Updated QA route: embeddings are generated on-demand
@app.route('/qa', methods=['GET', 'POST'])
def qa():
    ticket_id = request.args.get('ticket')
    if not ticket_id:
        return "No ticket id provided", 400

    # Generate QA embeddings on demand if not already processed.
    if ticket_id not in qa_data:
        process_ticket_for_qa(ticket_id)
    
    if ticket_id not in qa_data:
        return "No QA data available for this ticket.", 404

    result = None
    if request.method == 'POST':
        query = request.form.get('query')
        query_embedding = get_embedding(query).astype('float32')
        index = qa_data[ticket_id]['index']
        texts = qa_data[ticket_id]['texts']
        k = 3  # number of nearest neighbors
        distances, indices = index.search(np.array([query_embedding]), k)
        retrieved_texts = "\n\n".join([texts[i] for i in indices[0] if i < len(texts)])
        qa_prompt = f"Using the following context, answer the question:\nContext:\n{retrieved_texts}\n\nQuestion: {query}\nAnswer:"
        result = get_llm_response(qa_prompt)
    return render_template("qa.html", ticket_id=ticket_id, result=result)

# --------------------- Socket.IO Events --------------------- #
@socketio.on('join')
def handle_join(data):
    role = data.get('role')
    ticket_id = data.get('ticket_id')
    if not ticket_id:
        return
    join_room(ticket_id)
    print(f"{role} joined ticket: {ticket_id}")
    if ticket_id in active_modes:
        emit('mode_changed', {'mode': active_modes[ticket_id]}, to=request.sid)

@socketio.on('set_mode')
def handle_set_mode(data):
    ticket_id = data.get('ticket_id')
    mode = data.get('mode')
    active_modes[ticket_id] = mode
    emit('mode_changed', {'mode': mode}, room=ticket_id)

@socketio.on('chat_message')
def handle_chat_message(data):
    role = data.get('role')
    ticket_id = data.get('ticket_id')
    message = data.get('message')
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'role': role,
        'message': message,
        'ticket_id': ticket_id
    }
    conversation_logs.append(log_entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(conversation_logs, f)
    emit('chat_message', {'message': message, 'sender': role, 'ticket_id': ticket_id},
         room=ticket_id, include_self=False)

@socketio.on('audio_message')
def handle_audio_message(data):
    role = data.get('role')
    ticket_id = data.get('ticket_id')
    audio = data.get('audio')
    timestamp = datetime.now().isoformat()
    
    wav_filepath = None
    try:
        header, encoded = audio.split(",", 1)
        audio_data = base64.b64decode(encoded)
        temp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_webm.write(audio_data)
        temp_webm.close()
        
        wav_dir = "audio_files"
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
        file_ts = int(datetime.now().timestamp())
        filename = f"{ticket_id}_{role}_{file_ts}.wav"
        wav_filepath = os.path.join(wav_dir, filename)
        
        input_container = av.open(temp_webm.name)
        output_container = av.open(wav_filepath, mode='w', format='wav')
        input_stream = input_container.streams.audio[0]
        output_stream = output_container.add_stream("pcm_s16le", rate=input_stream.rate)
        
        for frame in input_container.decode(input_stream):
            frame.pts = None
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        output_container.close()
        input_container.close()
        os.remove(temp_webm.name)
    except Exception as e:
        wav_filepath = None
        print("Error converting audio to wav using PyAV:", e)
    
    recognized_text = ""
    detected_language = ""
    text_filepath = ""
    
    if wav_filepath:
        try:
            azure_audio_config = speechsdk.audio.AudioConfig(filename=wav_filepath)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect_source_language_config,
                audio_config=azure_audio_config
            )
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = result.text
                auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(result)
                detected_language = auto_detect_result.language
                print(f"Recognized speech: {recognized_text}")
                print(f"Detected language: {detected_language}")
            elif result.reason == speechsdk.ResultReason.NoMatch:
                recognized_text = "No speech could be recognized."
            elif result.reason == speechsdk.ResultReason.Canceled:
                recognized_text = "Speech recognition canceled or error."
        except Exception as e:
            recognized_text = f"Error during speech recognition: {e}"
    
    text_dir = os.path.join("text_files", ticket_id)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    text_filename = f"{role}_{file_ts}.txt"
    text_filepath = os.path.join(text_dir, text_filename)
    try:
        with open(text_filepath, 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"Detected Language: {detected_language}\nRecognized Text: {recognized_text}")
    except Exception as e:
        print("Error writing text file:", e)
    
    log_entry = {
        'timestamp': timestamp,
        'role': role,
        'audio': audio,
        'wav_file': wav_filepath,
        'recognized_text': recognized_text,
        'detected_language': detected_language,
        'text_file': text_filepath,
        'ticket_id': ticket_id
    }
    conversation_logs.append(log_entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(conversation_logs, f)
    
    analysis_audio_url = ""
    agent_messages = []
    if recognized_text.strip():
        agent_state = {
            "input": recognized_text,
            "language": detected_language if detected_language else "unknown",
            "translated_text": None,
            "analysis": None,
            "sentiment_priority": None,
            "messages": [],
            "tool_result": ""
        }
        final_state = agent_workflow.invoke(agent_state)
        agent_messages = [msg.content for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        analysis_text = final_state.get("analysis", "")
        
        if analysis_text:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            synthesis_result = synthesizer.speak_text_async(analysis_text).get()
            if synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = synthesis_result.audio_data
                output_dir = "synthesized_audio"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                file_ts = int(datetime.now().timestamp())
                audio_filename = f"{ticket_id}_{role}_{file_ts}_analysis.wav"
                output_path = os.path.join(output_dir, audio_filename)
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                print(f"Synthesized audio saved to {output_path}")
                encoded_audio = base64.b64encode(audio_data).decode('utf-8')
                analysis_audio_url = f"data:audio/wav;base64,{encoded_audio}"
            else:
                print("Speech synthesis failed for analysis text.")
    
    final_audio = analysis_audio_url if analysis_audio_url else audio
    
    emit('audio_message', {
            'audio': final_audio,
            'sender': role,
            'ticket_id': ticket_id,
            'recognized_text': recognized_text,
            'detected_language': detected_language,
            'agent_messages': agent_messages,
            'analysis_audio': analysis_audio_url
         },
         room=ticket_id, include_self=False)

# --------------------- Agentic Workflow Definitions --------------------- #

def _call_llm(prompt: str, temperature: float = 0.5, max_tokens: int = 1024) -> str:
    try:
        completion = client.chat.completions.create(
            model=NVIDIA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=1,
            max_tokens=max_tokens,
            stream=True
        )
        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        return response_text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Error: LLM call failed."

NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = "#"
NVIDIA_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-instruct"

client = OpenAI(
    base_url=NVIDIA_API_BASE_URL,
    api_key=NVIDIA_API_KEY
)

@tool(description="Translates the provided text to English. If the text is a mix of languages, it cleans and corrects errors.")
def translate_to_english(text: str, source_language: str) -> str:
    if source_language == 'en':
        return text
    prompt = f"""
Translate the following text to English. The text may be in {source_language} or a mix of {source_language} and English.
Text ({source_language}): {text}
Translation (English):
"""
    translated_text = _call_llm(prompt, temperature=0.5, max_tokens=512)
    print(f"Translated to English: {translated_text} (from: {text})")
    return translated_text

@tool(description="Analyzes the text and returns a concise one or two line summary that highlights the main topic.")
def analyze_text(text: str) -> str:
    prompt = f"""
Analyze the text and return a concise one or two line summary in English that highlights the main topic without any extra text or symbols like **.
Text: {text}
Analysis:
"""
    analysis = _call_llm(prompt, temperature=0.6, max_tokens=256)
    print(f"Text Analysis: {analysis}")
    return analysis

@tool(description="Analyzes the text to determine its sentiment and priority, returning a valid JSON object.")
def analyze_sentiment_priority(text: str) -> str:
    prompt = f"""
Analyze the following text and determine its sentiment and priority. Return ONLY a valid JSON object.
Text: {text}
{{ "sentiment": "...", "priority": "..." }}
"""
    result_json = _call_llm(prompt, temperature=0.0, max_tokens=128)
    print(f"Sentiment and Priority Raw Result: {result_json}")
    try:
        result_json = result_json.strip()
        start = result_json.find("{")
        end = result_json.rfind("}") + 1
        if start == -1 or end == -1:
           raise json.JSONDecodeError("No valid JSON found", result_json, 0)
        result_json = result_json[start:end]
        result = json.loads(result_json)
        if "sentiment" not in result or result["sentiment"] not in ["positive", "negative", "neutral"]:
            raise ValueError(f"Invalid or missing sentiment: {result.get('sentiment')}")
        if "priority" not in result or result["priority"] not in ["high", "medium", "low"]:
            raise ValueError(f"Invalid or missing priority: {result.get('priority')}")
        return json.dumps(result)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing sentiment/priority JSON: {e}. Raw LLM output: {result_json}")
        return json.dumps({"sentiment": "neutral", "priority": "medium"})

tools = [translate_to_english, analyze_text, analyze_sentiment_priority]
tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    input: str
    language: Optional[str]
    translated_text: Optional[str]
    analysis: Optional[str]
    sentiment_priority: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    tool_result: str

def translation_node(state: AgentState) -> dict:
    if state['language'] and state['language'] not in ['en', 'unknown']:
        tool_invocation = ToolInvocation(tool="translate_to_english", tool_input={"text": state["input"], "source_language": state["language"]})
        tool_result = tool_executor.invoke(tool_invocation)
        updated_state = {"translated_text": tool_result,
                         "messages": state["messages"] + [AIMessage(content=f"Translated: {tool_result}")] }
    elif state['language'] == 'unknown':
        updated_state = {"translated_text": "Could not translate",
                         "messages": state["messages"] + [AIMessage(content="Could not detect language, no translation available")]}
    else:
        updated_state = {"translated_text": state["input"],
                         "messages": state["messages"] + [AIMessage(content="No translation needed.")]}
    return updated_state

def analysis_node(state: AgentState) -> dict:
    text_to_analyze = state.get("translated_text", state["input"])
    tool_invocation = ToolInvocation(tool="analyze_text", tool_input={"text": text_to_analyze})
    tool_result = tool_executor.invoke(tool_invocation)
    updated_state = {"analysis": tool_result,
                     "messages": state["messages"] + [AIMessage(content=f"Analysis: {tool_result}")] }
    return updated_state

def sentiment_priority_node(state: AgentState) -> dict:
    text_to_analyze = state.get("translated_text", state["input"])
    tool_invocation = ToolInvocation(tool="analyze_sentiment_priority", tool_input={"text": text_to_analyze})
    tool_result = tool_executor.invoke(tool_invocation)
    updated_state = {"sentiment_priority": tool_result,
                     "messages": state["messages"] + [AIMessage(content=f"Sentiment and Priority: {tool_result}")] }
    return updated_state

workflow = StateGraph(AgentState)
workflow.add_node("translation", translation_node)
workflow.add_node("text_analysis", analysis_node)
workflow.add_node("sentiment_analysis", sentiment_priority_node)
workflow.set_entry_point("translation")
workflow.add_edge("translation", "text_analysis")
workflow.add_edge("text_analysis", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", END)
agent_workflow = workflow.compile()

# --------------------- Followup Agent Using LLM --------------------- #
def followup_agent(ticket_id: str) -> None:
    text_dir = os.path.join("text_files", ticket_id)
    combined_text = ""
    
    if os.path.exists(text_dir):
        for filename in os.listdir(text_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(text_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        combined_text += file_content + "\n"
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    else:
        print(f"No text files found for ticket_id: {ticket_id}")
        return

    if combined_text.strip():
        prompt = f"""
You are an expert conversation analyst tasked with identifying key followup actions.
Based on the conversation text below, generate a concise followup summary that highlights any recommended next steps or key insights that require further attention.
Conversation text:
{combined_text}

Followup Summary:
"""
        followup_analysis = _call_llm(prompt, temperature=0.6, max_tokens=512)
        print(f"Followup LLM analysis for ticket {ticket_id}:\n{followup_analysis}")
        
        timestamp = datetime.now().isoformat()
        followup_log_entry = {
            'timestamp': timestamp,
            'role': "followup_agent",
            'ticket_id': ticket_id,
            'followup_analysis': followup_analysis
        }
        conversation_logs.append(followup_log_entry)
        with open(LOG_FILE, 'w') as f:
            json.dump(conversation_logs, f)
        
        socketio.emit('followup_analysis', {
            'ticket_id': ticket_id,
            'analysis': followup_analysis
        }, room=ticket_id)
    else:
        print(f"No content to analyze for ticket_id: {ticket_id}")

# --------------------- Process QA Data --------------------- #
def process_ticket_for_qa(ticket_id: str) -> None:
    text_dir = os.path.join("text_files", ticket_id)
    if not os.path.exists(text_dir):
        print(f"No text files for QA processing for ticket_id: {ticket_id}")
        return
    embeddings, texts, file_names = create_embeddings_from_folder(text_dir)
    if embeddings:
        index = create_faiss_index(embeddings)
        qa_data[ticket_id] = {"index": index, "texts": texts, "file_names": file_names}
        print(f"QA data processed for ticket {ticket_id}: {len(embeddings)} embeddings created.")
    else:
        print(f"No embeddings generated for ticket_id: {ticket_id}")

# --------------------- Updated End Conversation Handler --------------------- #
@socketio.on('end_conversation')
def handle_end_conversation(data):
    ticket_id = data.get('ticket_id')
    if ticket_id in active_tickets:
        active_tickets.remove(ticket_id)
    if ticket_id in active_modes:
        active_modes.pop(ticket_id)
    emit('conversation_ended', {'ticket_id': ticket_id}, room=ticket_id)
    print(f"Conversation {ticket_id} ended.")

    # Trigger followup analysis immediately after conversation ends.
    followup_agent(ticket_id)
    # Removed process_ticket_for_qa(ticket_id) so QA embeddings are generated on-demand.

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
