from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from assign import (
    load_tickets_from_file,
    build_inverted_index,
    train_tfidf,
    simple_preprocess,
    normalize_tokens,
)

app = Flask(__name__)

# ── Load & index tickets once at startup ──────────────────────────────────────
TICKETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.txt")

if not os.path.exists(TICKETS_FILE):
    raise FileNotFoundError(f"tickets.txt not found at: {TICKETS_FILE}")

tickets = load_tickets_from_file(TICKETS_FILE)

if not tickets:
    raise ValueError("No tickets loaded. Check tickets.txt format: ID|description|solution")

inverted_index = build_inverted_index(tickets)
corpus = [t["description"] for t in tickets]
vectorizer, tfidf_matrix = train_tfidf(corpus)

print(f"[OK] Loaded {len(tickets)} tickets and built TF-IDF index.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", ticket_count=len(tickets))


@app.route("/search", methods=["POST"])
def search_endpoint():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Phase 1 — Query Analysis
    query_clean = simple_preprocess(query)
    all_tokens  = query_clean.split()
    triples     = normalize_tokens(all_tokens)
    stopped     = [t for t in all_tokens if t not in [p[0] for p in triples]]

    analysis = {
        "all_tokens": all_tokens,
        "stopped": stopped,
        "triples": [[orig, lemma, root] for orig, lemma, root in triples],
    }

    query_roots  = [root for _, _, root in triples]
    valid_tokens = [t for t in query_roots if t in vectorizer.vocabulary_]

    if not valid_tokens:
        return jsonify({"analysis": analysis, "results": []})

    # Phase 2 — Candidate retrieval via inverted index
    candidate_indices = set()
    for token in valid_tokens:
        candidate_indices.update(inverted_index.get(token, []))
    candidate_indices = sorted(candidate_indices)

    # Phase 3 — Cosine similarity ranking
    query_tfidf  = vectorizer.transform([query_clean])
    q_vec_dense  = query_tfidf.toarray()[0]
    term_indices = [vectorizer.vocabulary_[t] for t in valid_tokens]
    cand_tfidf   = tfidf_matrix[candidate_indices]
    sim_scores   = cos_sim(query_tfidf, cand_tfidf)[0]
    scored = sorted(zip(candidate_indices, sim_scores), key=lambda x: -x[1])[:top_k]

    # Phase 4 — Build response
    results = []
    for rank, (doc_idx, score) in enumerate(scored, 1):
        ticket      = tickets[doc_idx]
        d_vec_dense = tfidf_matrix[doc_idx].toarray()[0]
        sim_rows = []
        for t, idx in zip(valid_tokens, term_indices):
            q_val = float(q_vec_dense[idx])
            d_val = float(d_vec_dense[idx])
            if q_val > 0 or d_val > 0:
                sim_rows.append({"token": t, "q": q_val, "d": d_val})

        results.append({
            "rank": rank,
            "ticket_id": ticket["id"],
            "score": round(float(score), 4),
            "description": ticket["description"],
            "solution": ticket.get("solution", ""),
            "sim_rows": sim_rows,
        })

    return jsonify({"analysis": analysis, "results": results})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
