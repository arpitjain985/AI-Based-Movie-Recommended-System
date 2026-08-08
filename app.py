#Movie Recommender System — production-ready Streamlit app.


from __future__ import annotations

import os
import io
import json
import time
import pickle
import difflib
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry
from supabase import create_client, Client
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_secret(key: str, default: str = "") -> str:
    """
    Read a config value from either st.secrets (local secrets.toml /
    Streamlit Community Cloud) or an environment variable (Render, Docker,
    any other host). st.secrets itself raises if no secrets.toml exists
    anywhere on disk, so this is wrapped defensively rather than using
    st.secrets.get(), which would crash on hosts that only use env vars.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


TMDB_API_KEY = get_secret("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
BACKDROP_BASE = "https://image.tmdb.org/t/p/original"
PLACEHOLDER = "https://placehold.co/500x750/141414/e50914?text=No+Poster"

# A single pooled session with retry/backoff, reused across every request.
# This fixes the repeated SSLError/connection churn from creating a fresh
# `requests` call (and no session) on every single poster fetch.
_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 503, 504),
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))

# --------------------------------------------------------------------------
# Supabase — auth (incl. guest/anonymous) + persistent DB for favorites,
# watchlist, and search history.
# --------------------------------------------------------------------------
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY)


@st.cache_resource
def get_supabase() -> "Client | None":
    if not SUPABASE_ENABLED:
        return None
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _authed_client():
    """
    A Supabase client with the current user's access token attached to
    PostgREST calls, so row-level security scopes reads/writes to them.
    Falls back to the anon client (no rows visible) if nobody's signed in.
    """
    client = get_supabase()
    if client is None:
        return None
    sess = st.session_state.get("sb_session")
    if sess:
        client.postgrest.auth(sess["access_token"])
    return client


def _store_session(auth_response, is_anonymous: bool):
    user = auth_response.user
    session = auth_response.session
    st.session_state["sb_session"] = {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "user_id": user.id,
        "email": user.email,
        "is_anonymous": is_anonymous,
    }


def sign_up(email: str, password: str) -> str | None:
    """Returns an error message, or None on success."""
    client = get_supabase()
    try:
        res = client.auth.sign_up({"email": email, "password": password})
        if res.session is None:
            return "Check your inbox to confirm your email, then log in."
        _store_session(res, is_anonymous=False)
        return None
    except Exception as e:
        return str(e)


def sign_in(email: str, password: str) -> str | None:
    client = get_supabase()
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        _store_session(res, is_anonymous=False)
        return None
    except Exception as e:
        return str(e)


def sign_in_guest() -> str | None:
    client = get_supabase()
    try:
        res = client.auth.sign_in_anonymously()
        _store_session(res, is_anonymous=True)
        return None
    except Exception as e:
        return str(e)


def upgrade_guest(email: str, password: str) -> str | None:
    """Link an email/password to the current anonymous account so its
    favorites/watchlist/history carry over instead of being lost."""
    client = _authed_client()
    try:
        client.auth.update_user({"email": email, "password": password})
        sess = st.session_state.get("sb_session", {})
        sess["email"] = email
        sess["is_anonymous"] = False
        st.session_state["sb_session"] = sess
        return None
    except Exception as e:
        return str(e)


def sign_out():
    client = _authed_client()
    try:
        if client:
            client.auth.sign_out()
    except Exception:
        pass
    st.session_state["sb_session"] = None
    for k in ("favorites_cache", "watchlist_cache", "history_cache"):
        st.session_state.pop(k, None)


def current_user_id() -> str | None:
    sess = st.session_state.get("sb_session")
    return sess["user_id"] if sess else None


# --- Favorites / watchlist / history: DB reads+writes scoped by RLS to the
# current user_id. Each is mirrored into session_state as a cache so a
# rerun doesn't re-hit the DB for every card render. ---------------------
def _db_list(table: str, cache_key: str) -> list:
    uid = current_user_id()
    if not uid:
        return []
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    client = _authed_client()
    try:
        res = client.table(table).select("*").eq("user_id", uid).order("added_at", desc=True).execute()
        st.session_state[cache_key] = res.data
        return res.data
    except Exception:
        return []


def _db_add(table: str, cache_key: str, movie: dict):
    uid = current_user_id()
    if not uid:
        return
    client = _authed_client()
    try:
        client.table(table).upsert(
            {
                "user_id": uid,
                "movie_id": movie["movie_id"],
                "title": movie["title"],
                "poster": movie.get("poster"),
            },
            on_conflict="user_id,movie_id",
        ).execute()
        st.session_state.pop(cache_key, None)
    except Exception as e:
        st.toast(f"Couldn't save: {e}", icon="⚠️")


def _db_remove(table: str, cache_key: str, movie_id: int):
    uid = current_user_id()
    if not uid:
        return
    client = _authed_client()
    try:
        client.table(table).delete().eq("user_id", uid).eq("movie_id", movie_id).execute()
        st.session_state.pop(cache_key, None)
    except Exception as e:
        st.toast(f"Couldn't remove: {e}", icon="⚠️")


def get_favorites() -> list:
    return _db_list("favorites", "favorites_cache")


def add_favorite(movie: dict):
    _db_add("favorites", "favorites_cache", movie)


def remove_favorite(movie_id: int):
    _db_remove("favorites", "favorites_cache", movie_id)


def get_watchlist() -> list:
    return _db_list("watchlist", "watchlist_cache")


def add_watchlist(movie: dict):
    _db_add("watchlist", "watchlist_cache", movie)


def remove_watchlist(movie_id: int):
    _db_remove("watchlist", "watchlist_cache", movie_id)


def log_search(query: str):
    uid = current_user_id()
    if not uid or not query:
        return
    client = _authed_client()
    try:
        client.table("search_history").insert({"user_id": uid, "query": query}).execute()
        st.session_state.pop("history_cache", None)
    except Exception:
        pass


def get_history(limit: int = 10) -> list:
    cache_key = "history_cache"
    uid = current_user_id()
    if not uid:
        return []
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    client = _authed_client()
    try:
        res = (
            client.table("search_history")
            .select("*")
            .eq("user_id", uid)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        st.session_state[cache_key] = res.data
        return res.data
    except Exception:
        return []


# --------------------------------------------------------------------------
# Groq — natural-language search (LLM intent parsing) + voice search
# (Whisper transcription). Both reuse the TMDB/discover plumbing above.
# --------------------------------------------------------------------------
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_ENABLED = bool(GROQ_API_KEY)
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"


@st.cache_resource
def get_groq() -> "Groq | None":
    if not GROQ_ENABLED:
        return None
    return Groq(api_key=GROQ_API_KEY)


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """Speech-to-text via Groq-hosted Whisper. Returns None on failure."""
    client = get_groq()
    if client is None or not audio_bytes:
        return None
    try:
        result = client.audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=("query.wav", audio_bytes, "audio/wav"),
            response_format="text",
        )
        # SDK returns a plain string when response_format="text"
        return str(result).strip() or None
    except Exception as e:
        st.toast(f"Voice transcription failed: {e}", icon="⚠️")
        return None


def parse_query_with_llm(query: str, known_genres: list, known_titles: list) -> dict:
    """
    Ask the LLM to turn free text into structured intent:
    - reference_movie: a title the user is comparing to, if any
    - genres: subset of known_genres that fit
    - mood: one short word/phrase describing tone
    - summary: one sentence capturing what they're after (used for the reply)
    Falls back to an empty/neutral structure on any failure.
    """
    empty = {"reference_movie": None, "genres": [], "mood": None, "summary": query}
    client = get_groq()
    if client is None:
        return empty

    # Give the model a slice of real titles so it can recognize references
    # even to movies outside common knowledge, without shipping the whole catalog.
    sample_titles = ", ".join(known_titles[:150])

    system_prompt = f"""You turn a movie request into structured JSON. Output ONLY valid JSON, no prose, matching exactly:
{{"reference_movie": string or null, "genres": array of strings (choose only from the allowed list), "mood": string or null, "summary": string}}

Allowed genres: {", ".join(known_genres)}
Some titles that exist in the catalog (for reference matching, not exhaustive — the real movie may not be in this sample): {sample_titles}

Rules:
- reference_movie: the title the user explicitly compares to or names, else null. Use your best guess at the real, correctly spelled title.
- genres: only pick from the allowed list, 0-3 items, based on stated or implied genre.
- mood: a short phrase like "emotional", "lighthearted", "dark and tense", or null.
- summary: one friendly sentence restating what they want, to show back to the user."""

    try:
        resp = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_completion_tokens=300,
        )
        parsed = json.loads(resp.choices[0].message.content)
        parsed.setdefault("reference_movie", None)
        parsed.setdefault("genres", [])
        parsed.setdefault("mood", None)
        parsed.setdefault("summary", query)
        return parsed
    except Exception:
        return empty


def match_catalog_title(reference: str | None, titles: list, cutoff: float = 0.6) -> str | None:
    """Fuzzy-match an LLM-guessed title against the real catalog, since the
    model may slightly misspell or paraphrase it."""
    if not reference:
        return None
    matches = difflib.get_close_matches(reference, titles, n=1, cutoff=cutoff)
    return matches[0] if matches else None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_discover(genre_ids: tuple, k: int = 10) -> list:
    """TMDB discover, filtered by genre IDs, sorted by rating (with a vote
    floor so obscure/low-vote titles don't dominate)."""
    if not TMDB_API_KEY:
        return []
    params = {
        "api_key": TMDB_API_KEY,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 200,
        "language": "en-US",
    }
    if genre_ids:
        params["with_genres"] = ",".join(str(g) for g in genre_ids)
    try:
        resp = _session.get(f"{TMDB_BASE}/discover/movie", params=params, timeout=6)
        resp.raise_for_status()
        results = resp.json().get("results", [])[:k]
        return [
            {
                "movie_id": m["id"],
                "title": m["title"],
                "poster": f"{IMG_BASE}{m['poster_path']}" if m.get("poster_path") else PLACEHOLDER,
                "rating": m.get("vote_average"),
                "year": (m.get("release_date") or "")[:4],
                "genre_ids": m.get("genre_ids", []),
            }
            for m in results
        ]
    except requests.exceptions.RequestException:
        return []


def nl_search(query: str, movies: pd.DataFrame, similarity, region: str, genre_map: dict, k: int = 10):
    """
    Natural-language search entry point: parse intent, then either
    (a) run the existing similarity-based recommend() off a matched
        reference movie, or (b) fall back to a genre-filtered TMDB discover.
    Returns (results list, reply text).
    """
    known_genres = list(genre_map.values())
    known_titles = list(movies["title"].values)
    parsed = parse_query_with_llm(query, known_genres, known_titles)

    matched_title = match_catalog_title(parsed.get("reference_movie"), known_titles)

    if matched_title:
        results = recommend(movies, similarity, matched_title, k=k, region=region)
        reply = f"{parsed.get('summary', query)} Since **{matched_title}** is the closest match in the catalog, here's what's similar to it."
    else:
        genre_ids = tuple(
            sorted(
                gid for gid, name in genre_map.items() if name in (parsed.get("genres") or [])
            )
        )
        results = fetch_discover(genre_ids, k=k)
        if genre_ids:
            genre_names = ", ".join(parsed["genres"])
            reply = f"{parsed.get('summary', query)} Here are highly-rated {genre_names} picks."
        else:
            reply = f"{parsed.get('summary', query)} I couldn't pin down a specific genre or reference movie, so here's what's trending highly rated overall."
            results = fetch_discover((), k=k)

    return results, reply


# --------------------------------------------------------------------------
# Styling — dark, cinematic, Netflix-inspired but with its own identity
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(ellipse at top, #1a1a1a 0%, #0a0a0a 60%);
            color: #f5f5f5;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .cm-hero {
            padding: 2.2rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.6rem;
        }
        .cm-title {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3.4rem;
            letter-spacing: 0.04em;
            color: #ffffff;
            margin: 0;
            line-height: 1;
        }
        .cm-title span { color: #e50914; }
        .cm-subtitle {
            color: #9a9a9a;
            font-size: 0.98rem;
            margin-top: 0.35rem;
        }

        .cm-section-label {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.5rem;
            letter-spacing: 0.03em;
            color: #ffffff;
            margin: 1.6rem 0 0.8rem 0;
            border-left: 4px solid #e50914;
            padding-left: 0.6rem;
        }

        /* Movie card */
        .cm-card {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            background: #161616;
            border: 1px solid rgba(255,255,255,0.06);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .cm-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 16px 32px rgba(0,0,0,0.55);
            border-color: rgba(229,9,20,0.5);
        }
        .cm-poster-wrap { position: relative; width: 100%; aspect-ratio: 2/3; overflow: hidden; }
        .cm-poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .cm-rating-badge {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0,0,0,0.75);
            border: 1px solid rgba(255,255,255,0.15);
            color: #f5c518;
            font-weight: 700;
            font-size: 0.78rem;
            padding: 3px 7px;
            border-radius: 6px;
            backdrop-filter: blur(2px);
        }
        .cm-card-body { padding: 0.65rem 0.7rem 0.8rem 0.7rem; }
        .cm-card-title {
            font-weight: 600;
            font-size: 0.92rem;
            color: #fff;
            line-height: 1.25;
            margin-bottom: 2px;
            min-height: 2.4em;
        }
        .cm-card-meta {
            color: #8c8c8c;
            font-size: 0.78rem;
        }

        .cm-badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
        .cm-genre-pill {
            background: rgba(255,255,255,0.07);
            color: #cfcfcf;
            font-size: 0.68rem;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .cm-cast-row { display: flex; gap: 10px; overflow-x: auto; margin-top: 8px; padding-bottom: 4px; }
        .cm-cast-item { flex: 0 0 auto; width: 56px; text-align: center; }
        .cm-cast-item img {
            width: 48px; height: 48px; border-radius: 50%;
            object-fit: cover; border: 1px solid rgba(255,255,255,0.15);
        }
        .cm-cast-name { font-size: 0.62rem; color: #bdbdbd; margin-top: 3px; line-height: 1.1; }
        .cm-director { color: #9a9a9a; font-size: 0.78rem; margin-top: 4px; }

        .cm-provider-row { display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
        .cm-provider-row img {
            width: 30px; height: 30px; border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.15);
        }
        .cm-provider-label { color: #6f6f6f; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 2px; }

        div.stButton > button {
            background: #e50914;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
            transition: background 0.2s ease;
        }
        div.stButton > button:hover {
            background: #b0060f;
            color: white;
        }

        section[data-testid="stSidebar"] {
            background: #0d0d0d;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open("similarity.pkl", "rb"))
        return movies, similarity
    except FileNotFoundError as e:
        st.error(
            "Couldn't find `movies_dict.pkl` / `similarity.pkl` in this folder. "
            "Make sure both files sit next to app.py."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model data: {e}")
        st.stop()


# --------------------------------------------------------------------------
# TMDB helpers — cached so each movie is only fetched once per session,
# and results persist for an hour so a re-run doesn't re-hit the API.
# --------------------------------------------------------------------------
PROVIDER_LOGO_BASE = "https://image.tmdb.org/t/p/w92"
PROFILE_IMG_BASE = "https://image.tmdb.org/t/p/w185"
DEFAULT_REGION = "IN"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_genre_map() -> dict:
    """id -> name, used to render genre pills without a per-movie extra call."""
    if not TMDB_API_KEY:
        return {}
    try:
        resp = _session.get(
            f"{TMDB_BASE}/genre/movie/list",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=6,
        )
        resp.raise_for_status()
        return {g["id"]: g["name"] for g in resp.json().get("genres", [])}
    except requests.exceptions.RequestException:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_details(movie_id: int, region: str = DEFAULT_REGION) -> dict:
    """
    One call per movie (via append_to_response) covering:
    poster/rating/year/overview/genres, trailer, top cast + director,
    and where-to-watch providers for the given region.
    """
    default = {
        "poster": PLACEHOLDER,
        "backdrop": None,
        "rating": None,
        "year": None,
        "overview": "No overview available.",
        "genres": [],
        "genre_ids": [],
        "trailer_key": None,
        "cast": [],
        "director": None,
        "providers": {},
    }
    if not TMDB_API_KEY:
        return default

    url = f"{TMDB_BASE}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "append_to_response": "videos,credits,watch/providers",
    }
    try:
        resp = _session.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        poster_path = data.get("poster_path")
        backdrop_path = data.get("backdrop_path")
        release_date = data.get("release_date") or ""

        # Trailer: prefer an official YouTube "Trailer", fall back to "Teaser"
        videos = data.get("videos", {}).get("results", [])
        trailer = next(
            (v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"),
            next((v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Teaser"), None),
        )

        # Cast (top 5) + director
        credits = data.get("credits", {})
        cast = [
            {
                "name": c["name"],
                "character": c.get("character", ""),
                "photo": f"{PROFILE_IMG_BASE}{c['profile_path']}" if c.get("profile_path") else None,
            }
            for c in credits.get("cast", [])[:5]
        ]
        director = next(
            (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"), None
        )

        # Where to watch, scoped to region
        wp = data.get("watch/providers", {}).get("results", {}).get(region, {})
        providers = {
            kind: [
                {"name": p["provider_name"], "logo": f"{PROVIDER_LOGO_BASE}{p['logo_path']}"}
                for p in wp.get(kind, [])
            ]
            for kind in ("flatrate", "rent", "buy")
            if wp.get(kind)
        }

        return {
            "poster": f"{IMG_BASE}{poster_path}" if poster_path else PLACEHOLDER,
            "backdrop": f"{BACKDROP_BASE}{backdrop_path}" if backdrop_path else None,
            "rating": data.get("vote_average"),
            "year": release_date[:4] if release_date else None,
            "overview": data.get("overview") or "No overview available.",
            "genres": [g["name"] for g in data.get("genres", [])][:3],
            "genre_ids": [g["id"] for g in data.get("genres", [])],
            "trailer_key": trailer["key"] if trailer else None,
            "cast": cast,
            "director": director,
            "providers": providers,
        }
    except requests.exceptions.RequestException:
        return default


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending() -> list:
    """Trending-this-week rail for the landing view (basic card data only —
    full details are fetched lazily per-card via fetch_movie_details)."""
    if not TMDB_API_KEY:
        return []
    url = f"{TMDB_BASE}/trending/movie/week"
    try:
        resp = _session.get(url, params={"api_key": TMDB_API_KEY}, timeout=6)
        resp.raise_for_status()
        results = resp.json().get("results", [])[:10]
        return [
            {
                "movie_id": m["id"],
                "title": m["title"],
                "poster": f"{IMG_BASE}{m['poster_path']}" if m.get("poster_path") else PLACEHOLDER,
                "rating": m.get("vote_average"),
                "year": (m.get("release_date") or "")[:4],
                "genre_ids": m.get("genre_ids", []),
            }
            for m in results
        ]
    except requests.exceptions.RequestException:
        return []


def recommend(movies: pd.DataFrame, similarity, movie: str, k: int, region: str):
    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]
    ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : k + 1]

    results = []
    for i, score in ranked:
        movie_id = int(movies.iloc[i].movie_id)
        details = fetch_movie_details(movie_id, region=region)
        results.append(
            {
                "movie_id": movie_id,
                "title": movies.iloc[i].title,
                "match": round(score * 100),
                **details,
            }
        )
    return results


# --------------------------------------------------------------------------
# UI components
# --------------------------------------------------------------------------
def render_card(col, movie: dict, show_match: bool = False, region: str = "IN"):
    with col:
        rating_html = (
            f'<div class="cm-rating-badge">★ {movie["rating"]:.1f}</div>'
            if movie.get("rating")
            else ""
        )
        st.markdown(
            f"""
            <div class="cm-card">
                <div class="cm-poster-wrap">
                    <img src="{movie['poster']}" />
                    {rating_html}
                </div>
                <div class="cm-card-body">
                    <div class="cm-card-title">{movie['title']}</div>
                    <div class="cm-card-meta">
                        {movie.get('year') or ''}{' · ' + str(movie['match']) + '% match' if show_match and movie.get('match') is not None else ''}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        movie_id = movie.get("movie_id")
        # Trending-rail cards only carry basic fields until opened — fetch
        # the rich details (trailer/cast/providers) lazily on expand.
        needs_lazy_fetch = movie_id is not None and "trailer_key" not in movie

        with st.expander("Details"):
            full = fetch_movie_details(movie_id, region=region) if needs_lazy_fetch else movie

            if full.get("genres"):
                pills = "".join(f'<span class="cm-genre-pill">{g}</span>' for g in full["genres"])
                st.markdown(f'<div class="cm-badge-row">{pills}</div>', unsafe_allow_html=True)

            st.caption(full.get("overview", ""))

            # --- Trailer ---
            if full.get("trailer_key"):
                show_key = f"trailer_open_{movie_id}"
                if st.button("▶ Watch trailer", key=f"tr_{movie_id}_{movie['title']}"):
                    st.session_state[show_key] = True
                if st.session_state.get(show_key):
                    st.video(f"https://www.youtube.com/watch?v={full['trailer_key']}")
            else:
                st.caption("No trailer available.")

            # --- Cast & director ---
            if full.get("cast"):
                cast_html = "".join(
                    f"""<div class="cm-cast-item">
                            <img src="{c['photo'] or PLACEHOLDER}" />
                            <div class="cm-cast-name">{c['name']}</div>
                        </div>"""
                    for c in full["cast"]
                )
                st.markdown(f'<div class="cm-cast-row">{cast_html}</div>', unsafe_allow_html=True)
            if full.get("director"):
                st.markdown(f'<div class="cm-director">Directed by {full["director"]}</div>', unsafe_allow_html=True)

            # --- Where to watch ---
            providers = full.get("providers") or {}
            if providers:
                for kind, label in (("flatrate", "Stream"), ("rent", "Rent"), ("buy", "Buy")):
                    if providers.get(kind):
                        logos = "".join(f'<img src="{p["logo"]}" title="{p["name"]}" />' for p in providers[kind])
                        st.markdown(
                            f'<div class="cm-provider-row"><span class="cm-provider-label">{label}</span>{logos}</div>',
                            unsafe_allow_html=True,
                        )
            elif TMDB_API_KEY:
                st.caption(f"Not currently available on streaming in {region}.")

            # --- Favorites / watchlist ---
            if movie_id is not None:
                if not current_user_id():
                    st.caption("Sign in (sidebar) to save favorites or add to your watchlist.")
                else:
                    fav_ids = {f["movie_id"] for f in get_favorites()}
                    wl_ids = {w["movie_id"] for w in get_watchlist()}
                    c1, c2 = st.columns(2)
                    with c1:
                        if movie_id in fav_ids:
                            if st.button("♥ Favorited", key=f"fav_{movie_id}_{movie['title']}"):
                                remove_favorite(movie_id)
                                st.rerun()
                        else:
                            if st.button("♡ Favorite", key=f"fav_{movie_id}_{movie['title']}"):
                                add_favorite(movie)
                                st.rerun()
                    with c2:
                        if movie_id in wl_ids:
                            if st.button("✓ In watchlist", key=f"wl_{movie_id}_{movie['title']}"):
                                remove_watchlist(movie_id)
                                st.rerun()
                        else:
                            if st.button("+ Watchlist", key=f"wl_{movie_id}_{movie['title']}"):
                                add_watchlist(movie)
                                st.rerun()


def render_row(movies_list: list, show_match: bool = False, region: str = "IN"):
    cols = st.columns(len(movies_list)) if movies_list else []
    for col, m in zip(cols, movies_list):
        render_card(col, m, show_match=show_match, region=region)


def filter_by_genre(movies_list: list, genre_id) -> list:
    if not genre_id:
        return movies_list
    return [m for m in movies_list if genre_id in (m.get("genre_ids") or [])]


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
movies, similarity = load_data()

if not TMDB_API_KEY:
    st.warning(
        "No TMDB API key found — posters and metadata will show placeholders. "
        "Add `TMDB_API_KEY` to `.streamlit/secrets.toml` to enable them.",
        icon="⚠️",
    )

st.markdown(
    """
    <div class="cm-hero">
        <div class="cm-title">Cine<span>Match</span></div>
        <div class="cm-subtitle">Pick a movie you like — get recommendations tuned to it.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

genre_map = fetch_genre_map()  # id -> name

with st.sidebar:
    st.markdown("### 🎬 CineMatch")
    st.caption("Content-based movie recommender")
    st.divider()

    # ---------------- Auth ----------------
    if not SUPABASE_ENABLED:
        st.info(
            "Accounts aren't configured yet — add `SUPABASE_URL` and "
            "`SUPABASE_ANON_KEY` to `.streamlit/secrets.toml` to enable "
            "sign-in, favorites, and history.",
            icon="🔌",
        )
    else:
        sess = st.session_state.get("sb_session")
        if sess is None:
            st.markdown("**Account**")
            if st.button("👤 Continue as guest", use_container_width=True):
                err = sign_in_guest()
                if err:
                    st.error(err)
                else:
                    st.rerun()

            tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
            with tab_login:
                le = st.text_input("Email", key="login_email")
                lp = st.text_input("Password", type="password", key="login_pw")
                if st.button("Log in", key="login_btn"):
                    err = sign_in(le, lp)
                    if err:
                        st.error(err)
                    else:
                        st.rerun()
            with tab_signup:
                se = st.text_input("Email", key="signup_email")
                sp = st.text_input("Password", type="password", key="signup_pw")
                if st.button("Create account", key="signup_btn"):
                    err = sign_up(se, sp)
                    if err:
                        st.info(err) if "confirm" in err.lower() else st.error(err)
                    else:
                        st.rerun()
        else:
            if sess["is_anonymous"]:
                st.markdown("**Guest session**")
                st.caption("Your favorites are saved, but only for this account. Add an email to keep them permanently.")
                with st.expander("Save my account"):
                    ue = st.text_input("Email", key="upgrade_email")
                    up = st.text_input("Password", type="password", key="upgrade_pw")
                    if st.button("Save account", key="upgrade_btn"):
                        err = upgrade_guest(ue, up)
                        if err:
                            st.error(err)
                        else:
                            st.success("Account saved — check your inbox to confirm.")
                            st.rerun()
            else:
                st.markdown(f"**Signed in** · {sess['email']}")
            if st.button("Log out", use_container_width=True):
                sign_out()
                st.rerun()

    st.divider()

    region = st.selectbox(
        "Where-to-watch region",
        ["IN", "US", "GB", "CA", "AU"],
        index=0,
        help="Streaming availability shown in Details is scoped to this region.",
    )

    genre_options = ["All genres"] + [genre_map[g] for g in sorted(genre_map, key=lambda g: genre_map[g])]
    selected_genre_name = st.selectbox("Filter by genre", genre_options)
    selected_genre_id = None
    if selected_genre_name != "All genres":
        selected_genre_id = next(g for g, name in genre_map.items() if name == selected_genre_name)

    st.divider()

    # ---------------- Favorites / watchlist / history ----------------
    if current_user_id():
        favorites = get_favorites()
        st.markdown(f"**♥ Favorites ({len(favorites)})**")
        if favorites:
            for f in favorites[:8]:
                c1, c2 = st.columns([3, 1])
                c1.write(f"🎬 {f['title']}")
                if c2.button("✕", key=f"rmfav_{f['movie_id']}"):
                    remove_favorite(f["movie_id"])
                    st.rerun()
        else:
            st.caption("None yet — favorite a movie from its Details panel.")

        watchlist_rows = get_watchlist()
        st.markdown(f"**Watchlist ({len(watchlist_rows)})**")
        if watchlist_rows:
            for w in watchlist_rows[:8]:
                c1, c2 = st.columns([3, 1])
                c1.write(f"🎞️ {w['title']}")
                if c2.button("✕", key=f"rmwl_{w['movie_id']}"):
                    remove_watchlist(w["movie_id"])
                    st.rerun()
        else:
            st.caption("Nothing saved yet.")

        history = get_history(limit=8)
        if history:
            with st.expander(f"📈 Recent searches ({len(history)})"):
                for h in history:
                    st.caption(f"• {h['query']}")

        st.divider()

    st.caption(f"Catalog size: {len(movies):,} titles")
    st.caption(f"Session started {datetime.now().strftime('%H:%M')}")

col_select, col_k, col_btn = st.columns([3, 1, 1])
with col_select:
    selected_movie = st.selectbox("Choose a movie you enjoyed", movies["title"].values)
with col_k:
    num_recs = st.slider("How many", min_value=3, max_value=10, value=5)
with col_btn:
    st.write("")
    st.write("")
    go = st.button("Recommend", use_container_width=True)

# ---------------------------------------------------------------------
# 🤖 AI search — natural language + voice, powered by Groq
# ---------------------------------------------------------------------
st.markdown('<div class="cm-section-label">Ask CineMatch</div>', unsafe_allow_html=True)

if not GROQ_ENABLED:
    st.caption(
        "Natural-language and voice search aren't configured yet — add `GROQ_API_KEY` "
        "to `.streamlit/secrets.toml` to enable them."
    )
else:
    st.caption("Type or speak what you're in the mood for — e.g. \"something like Interstellar but more emotional\".")

    mic_col, text_col = st.columns([1, 3])
    with mic_col:
        audio_value = st.audio_input("🎙️ Speak", label_visibility="collapsed")
    with text_col:
        # If a new recording just came in, transcribe it once and stash the
        # text so the widget below picks it up as its starting value.
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            audio_hash = hash(audio_bytes)
            if st.session_state.get("last_audio_hash") != audio_hash:
                st.session_state["last_audio_hash"] = audio_hash
                with st.spinner("Transcribing…"):
                    transcript = transcribe_audio(audio_bytes)
                if transcript:
                    st.session_state["ai_query_input"] = transcript

        ai_query = st.text_input(
            "Describe what you want to watch",
            key="ai_query_input",
            label_visibility="collapsed",
            placeholder="I want a sci-fi movie like Interstellar with a strong emotional story",
        )

    ai_go = st.button("✨ Ask", key="ai_search_btn")

if go:
    st.session_state["result_mode"] = "classic"
    st.session_state["result_seed"] = selected_movie
    st.session_state["result_k"] = num_recs
elif GROQ_ENABLED and ai_go and st.session_state.get("ai_query_input"):
    st.session_state["result_mode"] = "ai"
    st.session_state["result_seed"] = st.session_state["ai_query_input"]
    st.session_state["result_k"] = 10

mode = st.session_state.get("result_mode")

if mode == "classic":
    seed = st.session_state["result_seed"]
    log_search(seed)
    with st.spinner("Finding movies you'll like…"):
        recommendations = recommend(movies, similarity, seed, k=st.session_state["result_k"], region=region)
    recommendations = filter_by_genre(recommendations, selected_genre_id)
    st.markdown(f'<div class="cm-section-label">Because you liked {seed}</div>', unsafe_allow_html=True)
    if not recommendations:
        st.info(f"No recommendations matched **{selected_genre_name}** — try a different genre or clear the filter.")
    else:
        for start in range(0, len(recommendations), 5):
            render_row(recommendations[start : start + 5], show_match=True, region=region)

elif mode == "ai":
    seed = st.session_state["result_seed"]
    log_search(seed)
    with st.spinner("Thinking about what fits…"):
        ai_results, reply = nl_search(seed, movies, similarity, region, genre_map, k=10)
    ai_results = filter_by_genre(ai_results, selected_genre_id)
    st.markdown('<div class="cm-section-label">CineMatch AI picks</div>', unsafe_allow_html=True)
    with st.chat_message("assistant"):
        st.write(reply)
    if not ai_results:
        st.info(f"No AI picks matched **{selected_genre_name}** — try a different genre or clear the filter.")
    else:
        for start in range(0, len(ai_results), 5):
            render_row(ai_results[start : start + 5], region=region)

else:
    trending = fetch_trending()
    trending = filter_by_genre(trending, selected_genre_id)
    if trending:
        st.markdown('<div class="cm-section-label">Trending this week</div>', unsafe_allow_html=True)
        for start in range(0, len(trending), 5):
            render_row(trending[start : start + 5], region=region)
    elif TMDB_API_KEY:
        st.info(f"Nothing trending matched **{selected_genre_name}** right now — try a different genre.")
    else:
        st.info("Pick a movie above and hit **Recommend** to get started.")
