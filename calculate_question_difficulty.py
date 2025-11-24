import os
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from openai import OpenAI
import ssl
import nltk

# Fix SSL certificate issue for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('cmudict', quiet=True)

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

YOU_API_KEY = os.environ.get("YOU_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CACHE_FILE = "you_cache.pkl"

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------
# LOAD / SAVE CACHE (pickle)
# -----------------------------------------------------

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        CACHE = pickle.load(f)
else:
    CACHE = {}

def save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(CACHE, f)


# -----------------------------------------------------
# YOU.COM REQUEST
# -----------------------------------------------------

def you_search(query: str):
    """Cached You.com API call."""
    if query in CACHE:
        return CACHE[query]

    if not YOU_API_KEY:
        raise ValueError("YOU_API_KEY environment variable is not set. Please set it before running.")

    # Initialize the SDK with your API key
    # you = youdotcom.You(YOU_API_KEY)

    # # Perform a search
    # results = you.search.unified(query=query)
    
    # # Cache the results
    # CACHE[query] = results
    # save_cache()
    
    # return results

# -----------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------

def embed(text: str):
    """Get embedding from OpenAI."""
    result = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(result.data[0].embedding)


# -----------------------------------------------------
# LLM SUMMARIZATION
# -----------------------------------------------------

def llm_summaries(snippets):
    """Summarize each snippet into a factual sentence."""
    if not snippets:
        return [""]

    summaries = []
    for snip in snippets:
        prompt = f"Summarize the following snippet into one factual, concise sentence:\n\n{snip}"

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        text = resp.output_text.strip()
        summaries.append(text)

    return summaries


# -----------------------------------------------------
# AGREEMENT: MEAN PAIRWISE COSINE SIMILARITY
# -----------------------------------------------------

def cosine_mean_pairwise(texts):
    if len(texts) <= 1:
        return 1.0

    embs = np.array([embed(t) for t in texts])
    sim = cosine_similarity(embs)

    # Mean of upper triangle (excluding diagonal)
    n = len(sim)
    vals = [sim[i, j] for i in range(n) for j in range(i+1, n)]
    return float(np.mean(vals))


# -----------------------------------------------------
# RELEVANCE: QUESTION ↔ SUMMARY SIMILARITY
# -----------------------------------------------------

def compute_relevance(question, summaries):
    q_emb = embed(question)
    s_embs = np.array([embed(s) for s in summaries])
    sims = cosine_similarity([q_emb], s_embs)[0]
    return float(np.mean(sims))


# -----------------------------------------------------
# TEXT COMPLEXITY
# -----------------------------------------------------

def avg_flesch(snippets):
    """Scaled complexity: convert Flesch Reading Ease to 0–1 difficulty."""
    if not snippets:
        return 0.5

    scores = []
    for s in snippets:
        try:
            fre = textstat.flesch_reading_ease(s)  # high = easier
        except:
            fre = 50
        # convert to difficulty (1 = hard, 0 = easy)
        diff = 1 - max(0, min(fre, 100)) / 100
        scores.append(diff)

    return float(np.mean(scores))


# -----------------------------------------------------
# RELATED QUESTIONS DISTANCE
# -----------------------------------------------------

def avg_embedding_distance(question, related_queries):
    if not related_queries:
        return 0.5

    q_emb = embed(question)
    r_embs = [embed(r) for r in related_queries]

    sims = cosine_similarity([q_emb], np.array(r_embs))[0]
    return float(1 - np.mean(sims))  # distance = 1 - similarity


# -----------------------------------------------------
# MAIN DIFFICULTY FUNCTION
# -----------------------------------------------------

def get_difficulty(question):
    # results = you_search(question)
    complexity = avg_flesch([question])
    return complexity

    # Extract web results from the SDK response
    # The response structure is: results.results.web (list of Web objects)
    web_results = []
    if hasattr(results, 'results') and hasattr(results.results, 'web'):
        web_results = results.results.web[:5]  # Get top 5 results
    
    # Extract snippets from web results
    # Each Web object has a 'snippets' attribute which is a list of strings
    snippets = []
    for web in web_results:
        if hasattr(web, 'snippets') and web.snippets:
            # Combine first 2 snippets from a single result
            snippet_text = ' '.join(web.snippets[:2])
            snippets.append(snippet_text)
        elif hasattr(web, 'description') and web.description:
            # Fallback to description if no snippets
            snippets.append(web.description)
    
    # Summaries from LLM
    summaries = llm_summaries(snippets)

    # Compute components
    agreement = cosine_mean_pairwise(summaries)
    relevance = compute_relevance(question, summaries)

    # Extract related queries if available
    related_queries = []
    if hasattr(results, 'results'):
        # Check for related queries in various possible locations
        if hasattr(results.results, 'related_queries') and results.results.related_queries:
            for q in results.results.related_queries:
                if hasattr(q, 'query'):
                    related_queries.append(q.query)
                elif isinstance(q, str):
                    related_queries.append(q)
        # Alternative: use news titles as related queries if available
        elif hasattr(results.results, 'news') and results.results.news:
            related_queries = [news.title for news in results.results.news[:5] if hasattr(news, 'title')]
    
    difficulty = (
        (1 - agreement)                / 3 +
        (1 - relevance)                / 3 +
        complexity                     / 3
    )

    return {
        # "difficulty": float(difficulty),
        # "agreement": agreement,
        # "relevance": relevance,
        "complexity": complexity,
    }


# -----------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------

if __name__ == "__main__":
    q = "What is the capital of France?"
    result = get_difficulty(q)
    print(result)
