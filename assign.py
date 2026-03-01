from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import numpy as np
import os

# ─────────────────────────────────────────────
# NLTK SETUP (stop words + lemmatizer + stemmer)
# ─────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    for _pkg, _folder in [("stopwords", "corpora"), ("wordnet", "corpora"),
                           ("omw-1.4", "corpora"), ("averaged_perceptron_tagger", "taggers"),
                           ("averaged_perceptron_tagger_eng", "taggers")]:
        try:
            nltk.data.find(f"{_folder}/{_pkg}")
        except LookupError:
            nltk.download(_pkg, quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _stemmer    = PorterStemmer()
    _USE_NLTK   = True
except ImportError:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOP_WORDS = set(ENGLISH_STOP_WORDS)
    _lemmatizer = None
    _stemmer    = None
    _USE_NLTK   = False

# Extra words that standard stop word lists commonly miss
_EXTRA_STOP_WORDS = {
    # verb forms not in NLTK
    "did", "got", "went", "came", "said", "told", "made", "took",
    "used", "tried", "worked", "started", "stopped", "opened", "closed",
    "clicked", "pressed", "typed", "checked", "looked", "seemed",
    "appeared", "happened", "occurred", "noticed", "found", "seen",
    # filler / vague
    "like", "just", "thing", "things", "stuff", "bit", "lot", "lots",
    "kind", "sort", "way", "ways", "time", "times", "day", "days",
    "something", "anything", "everything", "nothing", "someone",
    "anyone", "everyone", "somehow", "somewhere", "sometime",
    # discourse markers
    "also", "though", "even", "still", "already", "always", "never",
    "often", "usually", "sometimes", "maybe", "perhaps", "probably",
    "basically", "actually", "literally", "suddenly", "recently",
    "currently", "generally", "normally", "simply", "quite", "really",
    "very", "pretty", "rather", "mostly", "nearly", "almost", "enough",
    # support-ticket fillers
    "hi", "hello", "hey", "thanks", "thank", "please", "help",
    "issue", "problem", "ticket", "request", "question",
}
_STOP_WORDS |= _EXTRA_STOP_WORDS

# Common English contractions -> expanded forms
_CONTRACTIONS = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it's": "it is", "let's": "let us",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "we'd": "we would", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have",
}


# ─────────────────────────────────────────────
# LOAD TICKETS FROM TEXT FILE
# ─────────────────────────────────────────────
def load_tickets_from_file(filepath):
    tickets = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                print(f"  [WARN] Line {line_num}: expected 3 pipe-separated fields, got {len(parts)} -- skipped.")
                continue
            ticket_id   = parts[0].strip()
            description = parts[1].strip()
            solution    = "|".join(parts[2:]).strip()
            if not ticket_id or not description or not solution:
                print(f"  [WARN] Line {line_num}: one or more empty fields -- skipped.")
                continue
            tickets.append({"id": ticket_id, "description": description, "solution": solution})
    return tickets


# ─────────────────────────────────────────────
# CORE NLP / IR UTILITIES
# ─────────────────────────────────────────────
def simple_preprocess(text):
    """Lowercase, expand contractions, remove non-alphanumeric characters."""
    text = text.lower()
    for contraction, expansion in _CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def _get_wordnet_pos(word):
    """Map POS tag to a form WordNetLemmatizer accepts."""
    try:
        from nltk import pos_tag
        from nltk.corpus import wordnet
        tag = pos_tag([word])[0][1][0].upper()
        tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB,
                   "N": wordnet.NOUN, "R": wordnet.ADV}
        return tag_map.get(tag, wordnet.NOUN)
    except Exception:
        return "v"


def normalize_tokens(tokens):
    """
    For each token:
      1. Skip if it is a stop word
      2. Lemmatize using POS-aware WordNet lemmatizer  (thinking -> think)
      3. Stem using Porter stemmer                      (thinking -> think -> think)
      4. Skip if the root is also a stop word
    Returns list of (original, lemma, root) triples.
    """
    triples = []
    for t in tokens:
        if t in _STOP_WORDS:
            continue
        if _USE_NLTK:
            pos   = _get_wordnet_pos(t)
            lemma = _lemmatizer.lemmatize(t, pos=pos)
            if lemma in _STOP_WORDS:
                continue
            root  = _stemmer.stem(lemma)
        else:
            lemma = t
            root  = t
        if root not in _STOP_WORDS:
            triples.append((t, lemma, root))
    return triples


def build_inverted_index(tickets):
    """Build inverted index using normalized root tokens."""
    index = defaultdict(list)
    for idx, ticket in enumerate(tickets):
        raw_tokens  = simple_preprocess(ticket["description"]).split()
        root_tokens = [root for _, _, root in normalize_tokens(raw_tokens)]
        for token in set(root_tokens):
            index[token].append(idx)
    return index


def train_tfidf(corpus):
    """Fit TfidfVectorizer on normalized root tokens."""
    def tokenize(text):
        return [root for _, _, root in normalize_tokens(simple_preprocess(text).split())]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        norm="l2",
        token_pattern=None,
        stop_words=None
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def print_divider(char="-", length=75):
    print(char * length)


def wrap_paragraph(text, width=75, indent=6):
    words = text.split()
    lines = []
    current = " " * indent
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = " " * indent + word
        else:
            current += (" " if current.strip() else "") + word
    if current.strip():
        lines.append(current)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SEARCH ENGINE
# ─────────────────────────────────────────────
def search(query, tickets, vectorizer, tfidf_matrix, inverted_index, top_k=1):
    query_clean = simple_preprocess(query)
    all_tokens  = query_clean.split()

    triples     = normalize_tokens(all_tokens)
    stopped     = [t for t in all_tokens if t not in [p[0] for p in triples]]
    query_roots = [root for _, _, root in triples]

    # ── PHASE 1 ──────────────────────────────────────────────────────────
    print_divider("=")
    print("  PHASE 1: QUERY ANALYSIS")
    print_divider("=")
    print(f"  Original Query      : {query}")
    print(f"  All Tokens          : {all_tokens}")
    if stopped:
        print(f"  Stop Words (removed): {stopped}")
    if triples:
        print()
        print(f"  {'Original':<18} | {'Lemma':<18} | {'Root (stem)'}")
        print("  " + "-" * 58)
        for orig, lemma, root in triples:
            print(f"  {orig:<18} | {lemma:<18} | {root}")
    print(f"\n  Normalized Query Tokens: {query_roots}\n")

    # ── PHASE 2 ──────────────────────────────────────────────────────────
    valid_tokens = [t for t in query_roots if t in vectorizer.vocabulary_]
    if not valid_tokens:
        print("  [!] No matching terms found in the document vocabulary.")
        print("      Try rephrasing with more specific technical terms.\n")
        return []

    candidate_indices = set()
    print_divider()
    print("  PHASE 2: INVERTED INDEX LOOKUP  (Candidate Retrieval)")
    print_divider()
    for token in valid_tokens:
        matching_indices = inverted_index.get(token, [])
        matching_ids = [tickets[i]["id"] for i in matching_indices]
        print(f"  Root: '{token}' -> Matched Tickets: {matching_ids}")
        candidate_indices.update(matching_indices)

    candidate_indices = sorted(list(candidate_indices))
    print(f"\n  -> Total Candidate Documents Selected: {len(candidate_indices)}\n")

    # ── PHASE 3 ──────────────────────────────────────────────────────────
    query_tfidf      = vectorizer.transform([query_clean])
    q_vec_dense      = query_tfidf.toarray()[0]
    term_indices     = [vectorizer.vocabulary_[t] for t in valid_tokens]
    candidates_tfidf = tfidf_matrix[candidate_indices]
    sim_scores       = cosine_similarity(query_tfidf, candidates_tfidf)[0]
    scored           = list(zip(candidate_indices, sim_scores))

    print_divider()
    print("  PHASE 3: COSINE SIMILARITY CALCULATIONS")
    print("  (All terms are root/stem forms; stop words excluded)")
    print_divider()

    results_data = []
    for doc_idx, score in scored:
        ticket      = tickets[doc_idx]
        d_vec_dense = tfidf_matrix[doc_idx].toarray()[0]

        print(f"\n  Ticket : {ticket['id']}")
        print(f"  {'Root Token':<18} | {'Query TF-IDF':>12} | {'Doc TF-IDF':>12}")
        print("  " + "-" * 48)

        dot_prod = 0
        for t, idx in zip(valid_tokens, term_indices):
            q_val = q_vec_dense[idx]
            d_val = d_vec_dense[idx]
            if q_val > 0 or d_val > 0:
                print(f"  {t:<18} | {q_val:>12.4f} | {d_val:>12.4f}")
                dot_prod += q_val * d_val

        print("  " + "-" * 48)
        norm_q = np.linalg.norm(q_vec_dense)
        norm_d = np.linalg.norm(d_vec_dense)
        print(f"  Cosine Sim = {dot_prod:.4f} / ({norm_q:.4f} x {norm_d:.4f}) = {score:.4f}")

        results_data.append({"doc_idx": doc_idx, "score": score, "ticket": ticket})

    # ── PHASE 4 ──────────────────────────────────────────────────────────
    results_data.sort(key=lambda x: x["score"], reverse=True)
    top_results = results_data[:top_k]

    print("\n")
    print_divider("=")
    print(f"  PHASE 4: BEST MATCH SOLUTION")
    print_divider("=")

    final_output = []
    for rank, r in enumerate(top_results, start=1):
        t = r["ticket"]
        s = r["score"]

        print(f"\n  +-- BEST MATCH  |  Ticket: {t['id']}  |  Score: {s:.4f}")
        print(f"  |")
        print(f"  |  SOLUTION:")
        print(wrap_paragraph(t.get("solution", "No solution available."), width=75, indent=6))
        print(f"  +" + "-" * 70)

        final_output.append({
            "rank": rank,
            "ticket_id": t["id"],
            "score": round(float(s), 4),
            "description": t["description"],
            "solution": t.get("solution", "No solution available.")
        })

    return final_output


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    TICKETS_FILE = os.path.join(SCRIPT_DIR, "tickets.txt")

    print("=" * 75)
    print("   TICKET SUPPORT SEARCH SYSTEM  --  TF-IDF + Inverted Index")
    print("=" * 75)
    print(f"\n  Looking for tickets file at:\n    {TICKETS_FILE}\n")

    if not os.path.exists(TICKETS_FILE):
        print("[ERROR] tickets.txt not found.")
        print(f"        Place 'tickets.txt' in the same folder as this script:")
        print(f"        {SCRIPT_DIR}")
        exit(1)

    tickets = load_tickets_from_file(TICKETS_FILE)

    if not tickets:
        print("[ERROR] No tickets were loaded.")
        print("        Each line in tickets.txt must follow this format:")
        print("        TICKET_ID|description paragraph|solution paragraph")
        exit(1)

    print(f"  Loaded {len(tickets)} tickets.")
    print("  Building inverted index ...")
    inverted_index = build_inverted_index(tickets)

    print("  Training TF-IDF vectorizer ...")
    corpus = [ticket["description"] for ticket in tickets]
    vectorizer, tfidf_matrix = train_tfidf(corpus)

    print(f"\n  System ready. {len(tickets)} tickets indexed.")
    print("  Describe your support issue in 1-3 sentences. The system will")
    print("  find the closest matching ticket and display its full solution.")
    print_divider("=")

    while True:
        try:
            print()
            user_query = input("  Describe your support issue (or type 'exit' to quit):\n  > ").strip()

            if user_query.lower() in ("exit", "quit", "q"):
                print("\n  Goodbye!\n")
                break

            if not user_query:
                print("  [!] Please enter a description of your support issue.")
                continue

            search(
                query=user_query,
                tickets=tickets,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                inverted_index=inverted_index,
                top_k=1
            )

        except KeyboardInterrupt:
            print("\n\n  Session interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  [ERROR] An unexpected error occurred: {e}\n")