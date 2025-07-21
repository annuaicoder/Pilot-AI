
from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, re
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

# === Math Parsing ===
def try_math(query):
    replacements = {
        "plus": "+", "add": "+", "added to": "+",
        "minus": "-", "subtract": "-", "subtracted from": "-",
        "times": "*", "x": "*", "multiplied by": "*",
        "divided by": "/", "over": "/", "÷": "/", "×": "*"
    }
    query = query.lower()
    for word, symbol in replacements.items():
        query = query.replace(word, symbol)
    match = re.search(r"([-+]?\d*\.?\d+)\s*([\+\-\*/])\s*([-+]?\d*\.?\d+)", query)
    if match:
        a, op, b = match.groups()
        try:
            result = eval(f"{float(a)}{op}{float(b)}")
            return f"{a} {op} {b} is {int(result) if result.is_integer() else result}."
        except:
            return None
    return None

# === Load and Embed Dataset ===
def load_examples(path):
    examples = []
    for fname in os.listdir(path):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)
                examples.extend(data if isinstance(data, list) else [data])
    return examples

examples = load_examples("data")
model = SentenceTransformer("all-MiniLM-L6-v2")
user_inputs = [e['user_input'] for e in examples]
embeddings = model.encode(user_inputs, convert_to_tensor=True)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("question", "")
    math_result = try_math(query)
    if math_result:
        return jsonify({"response": math_result})

    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score > 0.4:
        return jsonify({"response": examples[best_idx]['response']})
    return jsonify({"response": "🤖 Sorry, I couldn't quite understand. Try rephrasing your question."})

if __name__ == "__main__":
    app.run(debug=True)
