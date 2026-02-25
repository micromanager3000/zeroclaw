//! Gateway session management for multi-turn conversations.
//!
//! Each session gets a dedicated tokio task (worker) that owns the conversation
//! history and processes messages sequentially via an mpsc channel. This ensures
//! that concurrent messages to the same session are queued and processed in order,
//! each seeing the full prior context.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::config::schema::SessionConfig;
use crate::config::Config;
use crate::providers::ChatMessage;

/// Channel buffer size for session message queues.
const SESSION_CHANNEL_BUFFER: usize = 16;

/// A request sent to a session worker.
struct SessionRequest {
    message: String,
    response_tx: oneshot::Sender<Result<String>>,
}

/// Handle to a running session worker.
struct SessionHandle {
    tx: mpsc::Sender<SessionRequest>,
    last_active: Arc<Mutex<Instant>>,
    created_at: Instant,
    message_count: Arc<std::sync::atomic::AtomicUsize>,
}

/// Public info about a session (for list/get API).
#[derive(Debug, Clone, Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub message_count: usize,
    pub created_at_secs_ago: u64,
    pub last_active_secs_ago: u64,
}

/// Detailed session info including history (for get API).
#[derive(Debug, Clone, Serialize)]
pub struct SessionDetail {
    pub id: String,
    pub message_count: usize,
    pub created_at_secs_ago: u64,
    pub last_active_secs_ago: u64,
    pub history: Vec<SessionMessage>,
}

/// A single message in session history (serialized for API/persistence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
}

impl From<&ChatMessage> for SessionMessage {
    fn from(m: &ChatMessage) -> Self {
        Self {
            role: m.role.clone(),
            content: m.content.clone(),
        }
    }
}

/// Manages gateway sessions with per-session workers.
pub struct SessionManager {
    sessions: Mutex<HashMap<String, SessionHandle>>,
    /// Shared history snapshots for API reads. Workers update this after each turn.
    history_snapshots: Arc<Mutex<HashMap<String, Vec<SessionMessage>>>>,
    session_config: SessionConfig,
    app_config: Arc<Mutex<Config>>,
    persist_dir: Option<PathBuf>,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(
        session_config: SessionConfig,
        app_config: Arc<Mutex<Config>>,
        workspace_dir: &Path,
    ) -> Self {
        let persist_dir = if session_config.persist_to_file {
            let dir = workspace_dir.join(".sessions");
            if let Err(e) = std::fs::create_dir_all(&dir) {
                tracing::warn!("Failed to create sessions dir {}: {e}", dir.display());
            }
            Some(dir)
        } else {
            None
        };

        Self {
            sessions: Mutex::new(HashMap::new()),
            history_snapshots: Arc::new(Mutex::new(HashMap::new())),
            session_config,
            app_config,
            persist_dir,
        }
    }

    /// Send a message to a session and wait for the response.
    pub async fn send_message(&self, session_id: &str, message: String) -> Result<String> {
        let (response_tx, response_rx) = oneshot::channel();

        let tx = {
            let mut sessions = self.sessions.lock();
            let handle = sessions
                .entry(session_id.to_string())
                .or_insert_with(|| self.spawn_worker(session_id));
            *handle.last_active.lock() = Instant::now();
            handle.tx.clone()
        };

        let request = SessionRequest {
            message,
            response_tx,
        };

        tx.send(request)
            .await
            .map_err(|_| anyhow::anyhow!("Session worker closed unexpectedly"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Session worker dropped response channel"))?
    }

    /// List all active sessions.
    pub fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.lock();
        let now = Instant::now();
        let mut infos: Vec<SessionInfo> = sessions
            .iter()
            .map(|(id, handle)| {
                let last_active = *handle.last_active.lock();
                SessionInfo {
                    id: id.clone(),
                    message_count: handle
                        .message_count
                        .load(std::sync::atomic::Ordering::Relaxed),
                    created_at_secs_ago: now.duration_since(handle.created_at).as_secs(),
                    last_active_secs_ago: now.duration_since(last_active).as_secs(),
                }
            })
            .collect();
        infos.sort_by_key(|s| std::cmp::Reverse(s.last_active_secs_ago));
        infos
    }

    /// Get detailed info about a session including history.
    pub fn get_session(&self, session_id: &str) -> Option<SessionDetail> {
        let sessions = self.sessions.lock();
        let handle = sessions.get(session_id)?;
        let now = Instant::now();
        let last_active = *handle.last_active.lock();
        let history = self
            .history_snapshots
            .lock()
            .get(session_id)
            .cloned()
            .unwrap_or_default();

        Some(SessionDetail {
            id: session_id.to_string(),
            message_count: handle
                .message_count
                .load(std::sync::atomic::Ordering::Relaxed),
            created_at_secs_ago: now.duration_since(handle.created_at).as_secs(),
            last_active_secs_ago: now.duration_since(last_active).as_secs(),
            history,
        })
    }

    /// Delete a session and its persisted file.
    pub fn delete_session(&self, session_id: &str) -> bool {
        let removed = self.sessions.lock().remove(session_id).is_some();
        self.history_snapshots.lock().remove(session_id);
        if let Some(ref dir) = self.persist_dir {
            let path = dir.join(format!("{session_id}.json"));
            let _ = std::fs::remove_file(path);
        }
        removed
    }

    /// Start a background task that cleans up expired sessions.
    pub fn start_cleanup_task(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        let ttl = Duration::from_secs(self.session_config.ttl_hours * 3600);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // check every 5 min
            loop {
                interval.tick().await;
                let expired: Vec<String> = {
                    let sessions = manager.sessions.lock();
                    let now = Instant::now();
                    sessions
                        .iter()
                        .filter(|(_, handle)| {
                            now.duration_since(*handle.last_active.lock()) > ttl
                        })
                        .map(|(id, _)| id.clone())
                        .collect()
                };
                for id in &expired {
                    tracing::info!("Session {id} expired (TTL {}h)", manager.session_config.ttl_hours);
                    manager.delete_session(id);
                }
            }
        });
    }

    /// Spawn a new session worker task.
    fn spawn_worker(&self, session_id: &str) -> SessionHandle {
        let (tx, rx) = mpsc::channel(SESSION_CHANNEL_BUFFER);
        let last_active = Arc::new(Mutex::new(Instant::now()));
        let message_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let worker = SessionWorker {
            session_id: session_id.to_string(),
            history: load_persisted_history(session_id, self.persist_dir.as_deref()),
            rx,
            last_active: Arc::clone(&last_active),
            message_count: Arc::clone(&message_count),
            app_config: Arc::clone(&self.app_config),
            max_history: self.session_config.max_history_messages,
            persist_dir: self.persist_dir.clone(),
            history_snapshots: Arc::clone(&self.history_snapshots),
        };

        tokio::spawn(worker.run());

        SessionHandle {
            tx,
            last_active,
            created_at: Instant::now(),
            message_count,
        }
    }
}

/// Per-session worker that owns the conversation history.
struct SessionWorker {
    session_id: String,
    history: Vec<ChatMessage>,
    rx: mpsc::Receiver<SessionRequest>,
    last_active: Arc<Mutex<Instant>>,
    message_count: Arc<std::sync::atomic::AtomicUsize>,
    app_config: Arc<Mutex<Config>>,
    max_history: usize,
    persist_dir: Option<PathBuf>,
    history_snapshots: Arc<Mutex<HashMap<String, Vec<SessionMessage>>>>,
}

impl SessionWorker {
    async fn run(mut self) {
        tracing::info!("Session worker started: {}", self.session_id);

        while let Some(req) = self.rx.recv().await {
            *self.last_active.lock() = Instant::now();
            self.message_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Clone config snapshot for this turn
            let config = self.app_config.lock().clone();

            let result = crate::agent::process_message_with_history(
                config,
                &req.message,
                Some(&mut self.history),
            )
            .await;

            // Trim history to prevent unbounded growth
            crate::agent::trim_history(&mut self.history, self.max_history);

            // Update the shared snapshot for API reads
            {
                let snapshot: Vec<SessionMessage> =
                    self.history.iter().map(SessionMessage::from).collect();
                self.history_snapshots
                    .lock()
                    .insert(self.session_id.clone(), snapshot);
            }

            // Persist to file
            if let Some(ref dir) = self.persist_dir {
                persist_history(&self.session_id, dir, &self.history);
            }

            // Send response (ignore if receiver dropped)
            let _ = req.response_tx.send(result);
        }

        tracing::info!("Session worker stopped: {}", self.session_id);
    }
}

/// Load persisted session history from a JSON file.
fn load_persisted_history(session_id: &str, persist_dir: Option<&Path>) -> Vec<ChatMessage> {
    let dir = match persist_dir {
        Some(d) => d,
        None => return Vec::new(),
    };
    let path = dir.join(format!("{session_id}.json"));
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str::<Vec<ChatMessage>>(&data) {
            Ok(history) => {
                tracing::info!(
                    "Loaded session {session_id} with {} messages from {}",
                    history.len(),
                    path.display()
                );
                history
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to parse session file {}: {e}",
                    path.display()
                );
                Vec::new()
            }
        },
        Err(_) => Vec::new(), // File doesn't exist yet
    }
}

/// Persist session history to a JSON file.
fn persist_history(session_id: &str, dir: &Path, history: &[ChatMessage]) {
    let path = dir.join(format!("{session_id}.json"));
    match serde_json::to_string_pretty(history) {
        Ok(data) => {
            if let Err(e) = std::fs::write(&path, data) {
                tracing::warn!("Failed to persist session {session_id}: {e}");
            }
        }
        Err(e) => {
            tracing::warn!("Failed to serialize session {session_id}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_message_from_chat_message() {
        let chat = ChatMessage::user("hello");
        let session: SessionMessage = SessionMessage::from(&chat);
        assert_eq!(session.role, "user");
        assert_eq!(session.content, "hello");
    }

    #[test]
    fn load_persisted_history_returns_empty_for_missing_file() {
        let dir = std::env::temp_dir().join("zeroclaw_test_sessions_nonexistent");
        let history = load_persisted_history("no_such_session", Some(&dir));
        assert!(history.is_empty());
    }

    #[test]
    fn persist_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("zeroclaw_test_sessions_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();

        let history = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        persist_history("test_roundtrip", &dir, &history);
        let loaded = load_persisted_history("test_roundtrip", Some(&dir));

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].role, "system");
        assert_eq!(loaded[1].role, "user");
        assert_eq!(loaded[1].content, "hello");
        assert_eq!(loaded[2].role, "assistant");
        assert_eq!(loaded[2].content, "Hi there!");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_persisted_history_returns_empty_for_none_dir() {
        let history = load_persisted_history("any_id", None);
        assert!(history.is_empty());
    }
}
