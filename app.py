from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

# Available Groq models (fast inference)
GROQ_MODEL = "llama-3.3-70b-versatile"  # alternatives: mixtral-8x7b-32768, gemma2-9b-it

# ── LangChain chain (NO memory - stateless per request) ──────────────────────
def get_chain():
    load_dotenv(override=True)  # re-read .env on every request
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Create a .env file and add: GROQ_API_KEY=gsk_..."
        )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1024,
        groq_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful, friendly, and concise AI assistant powered by Groq. "
            "Answer the user's question clearly and accurately. "
            "Since you have NO memory, treat every message as a fresh conversation.",
        ),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        chain = get_chain()
        response = chain.invoke({"question": user_message})
        return jsonify({"response": response, "model": GROQ_MODEL})
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    api_key = os.getenv("GROQ_API_KEY")
    has_key = bool(api_key and api_key != "your_groq_api_key_here")
    return jsonify({"status": "ok", "provider": "groq", "model": GROQ_MODEL, "api_key_set": has_key})


if __name__ == "__main__":
    print("[LangChain + Groq Chatbot] No-memory mode running at http://127.0.0.1:5000")
    print("[!] Make sure you have a .env file with GROQ_API_KEY=gsk_...")
    print(f"[*] Using model: {GROQ_MODEL}")
    app.run(debug=True, port=5000)
