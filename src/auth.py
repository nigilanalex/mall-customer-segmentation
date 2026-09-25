"""Local demonstration authentication with hashed passwords and SQL sessions.

The shared demo account is not a production identity service. Session tokens
stay in Streamlit's server-side session state; no credentials are put in URLs.
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from .config import BASE

AUTH_DB = BASE / "database" / "auth.db"
ITERATIONS = 600_000
IDLE_SECONDS = 30 * 60
MAX_SECONDS = 8 * 60 * 60


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()


def connect(path=AUTH_DB):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt TEXT NOT NULL,
        password_hash TEXT NOT NULL, role TEXT NOT NULL, failures INTEGER DEFAULT 0, locked_until REAL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS auth_sessions(token_hash TEXT PRIMARY KEY, username TEXT NOT NULL,
        created REAL NOT NULL, last_seen REAL NOT NULL, FOREIGN KEY(username) REFERENCES users(username));
    """)
    if not connection.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        salt = secrets.token_hex(16)
        username = os.environ.get("AI_ADMIN_USERNAME", "admin")
        password = os.environ.get("AI_ADMIN_PASSWORD", "admin123")
        connection.execute("INSERT OR IGNORE INTO users(username,salt,password_hash,role) VALUES(?,?,?,?)",
                           (username, salt, password_hash(password, salt), "Administrator"))
        connection.commit()
    return connection


def authenticate(username, password, path=AUTH_DB, now=None):
    now = time.time() if now is None else now
    with closing(connect(path)) as connection, connection:
        row = connection.execute("SELECT salt,password_hash,role,failures,locked_until FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            password_hash(password, "00" * 16)  # Similar work for unknown usernames.
            return None
        salt, expected, role, failures, locked_until = row
        if locked_until > now:
            return None
        if not hmac.compare_digest(password_hash(password, salt), expected):
            failures = failures + 1 if locked_until == 0 else 1
            connection.execute("UPDATE users SET failures=?,locked_until=? WHERE username=?",
                               (failures, now + 60 if failures >= 5 else 0, username))
            return None
        connection.execute("UPDATE users SET failures=0,locked_until=0 WHERE username=?", (username,))
        token = secrets.token_urlsafe(32)
        connection.execute("INSERT INTO auth_sessions VALUES(?,?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), username, now, now))
        return token


def session_user(token, path=AUTH_DB, now=None):
    if not token:
        return None
    now = time.time() if now is None else now
    digest = hashlib.sha256(token.encode()).hexdigest()
    with closing(connect(path)) as connection, connection:
        row = connection.execute("SELECT s.username,u.role,s.created,s.last_seen FROM auth_sessions s JOIN users u ON u.username=s.username WHERE token_hash=?", (digest,)).fetchone()
        if not row:
            return None
        username, role, created, last_seen = row
        if now - created >= MAX_SECONDS or now - last_seen >= IDLE_SECONDS:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash=?", (digest,))
            return None
        connection.execute("UPDATE auth_sessions SET last_seen=? WHERE token_hash=?", (now, digest))
        return {"username": username, "role": role}


def revoke_session(token, path=AUTH_DB):
    if token:
        with closing(connect(path)) as connection, connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))


def require_login():
    """Gate all model loading, uploads, navigation and page execution."""
    import streamlit as st
    user = session_user(st.session_state.get("auth_token"))
    if user:
        return user
    if "auth_token" in st.session_state:
        st.session_state.clear()
    st.set_page_config(page_title="Admin login | Customer intelligence")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.caption("AI CUSTOMER INTELLIGENCE")
        st.title("Customer analytics, ready for your next decision.")
        st.markdown("Explore customer behavior, compare machine learning models, and prepare personalized marketing recommendations.")
        with st.container(horizontal=True):
            st.metric("Demo customers", "2,000", border=True)
            st.metric("Clustering algorithms", "4", border=True)
        st.caption("College project demonstration · Synthetic data · Local administrator access")
    with right, st.container(border=True):
        st.subheader("Administrator login", icon=":material/lock:")
        st.caption("Sign in to open the analytics dashboard.")
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            token = authenticate(username.strip(), password)
            if token:
                st.session_state.auth_token = token
                st.rerun()
            st.error("Sign-in failed. Check your credentials or wait one minute if several attempts failed.")
        st.caption("Demo account: admin / admin123. Sessions expire after 30 minutes of inactivity.")
    st.stop()
