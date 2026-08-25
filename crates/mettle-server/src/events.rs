//! Profile lifecycle event bus (agent-docs/profiles-contract.md
//! "`GET /engine/events` — SSE, replays last 100 on connect").
//!
//! Every activation/degradation/slot transition is broadcast to live SSE
//! subscribers and kept in a bounded replay buffer so a consumer that
//! connects mid-activation still sees the full picture. Events are the ONLY
//! completion signal for `PUT /engine/profiles/active` (202 = accepted,
//! outcome arrives as [`EngineEvent::ProfileActive`] or
//! [`EngineEvent::ProfileDegraded`]).

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Replay buffer cap (contract: "replays last 100 on connect").
const REPLAY_CAP: usize = 100;
/// Broadcast channel capacity. A slow subscriber that falls further behind
/// than this gets `Lagged` and simply skips ahead (SSE consumers tolerate
/// gaps; the replay buffer is the durability story).
const CHANNEL_CAP: usize = 256;

/// One profile-system lifecycle event. Serialized with an internally tagged
/// `"event"` field whose value is the contract's dotted name
/// (`profile.activating`, `slot.loaded`, ...).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event")]
pub enum EngineEvent {
    #[serde(rename = "profile.activating")]
    ProfileActivating { name: String },
    #[serde(rename = "profile.active")]
    ProfileActive { name: String },
    /// Degraded-but-successful activation, or a refused one (`dropped`
    /// naming every slot given up on, `shrunk` every slot loaded below full
    /// context). Nothing degrades silently.
    #[serde(rename = "profile.degraded")]
    ProfileDegraded {
        name: String,
        dropped: Vec<String>,
        shrunk: Vec<String>,
        reason: String,
    },
    #[serde(rename = "slot.loaded")]
    SlotLoaded { id: String },
    #[serde(rename = "slot.unloaded")]
    SlotUnloaded { id: String },
    #[serde(rename = "slot.failed")]
    SlotFailed { id: String, reason: String },
    /// Free VRAM dipped below the planner margin while resident.
    #[serde(rename = "vram.pressure")]
    VramPressure { free_bytes: u64 },
}

impl EngineEvent {
    /// Contract wire name (`"profile.activating"`, ...).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ProfileActivating { .. } => "profile.activating",
            Self::ProfileActive { .. } => "profile.active",
            Self::ProfileDegraded { .. } => "profile.degraded",
            Self::SlotLoaded { .. } => "slot.loaded",
            Self::SlotUnloaded { .. } => "slot.unloaded",
            Self::SlotFailed { .. } => "slot.failed",
            Self::VramPressure { .. } => "vram.pressure",
        }
    }
}

/// Fan-out + replay hub shared via `AppState`.
pub struct EventBus {
    tx: broadcast::Sender<Arc<str>>,
    replay: Mutex<VecDeque<Arc<str>>>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(CHANNEL_CAP);
        Self {
            tx,
            replay: Mutex::new(VecDeque::with_capacity(REPLAY_CAP)),
        }
    }

    /// Publish to live subscribers and append to the replay buffer.
    pub fn emit(&self, event: EngineEvent) {
        let json: Arc<str> = match serde_json::to_string(&event) {
            Ok(s) => Arc::from(s.as_str()),
            Err(e) => {
                tracing::error!("event {:?} failed to serialize: {e}", event.kind());
                return;
            }
        };
        if let Ok(mut replay) = self.replay.lock() {
            replay.push_back(json.clone());
            while replay.len() > REPLAY_CAP {
                replay.pop_front();
            }
        }
        let _ = self.tx.send(json);
    }

    /// (replay buffer, live subscription) for an SSE handler: yield the
    /// replay first, then forward the broadcast stream.
    pub fn subscribe(&self) -> (Vec<Arc<str>>, broadcast::Receiver<Arc<str>>) {
        let replay = self
            .replay
            .lock()
            .map(|r| r.iter().cloned().collect())
            .unwrap_or_default();
        (replay, self.tx.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_serialize_with_dotted_tag() {
        let degraded = EngineEvent::ProfileDegraded {
            name: "finance".into(),
            dropped: vec!["big-model".into()],
            shrunk: vec!["qwen35".into()],
            reason: "vram".into(),
        };
        let json = serde_json::to_value(&degraded).unwrap();
        assert_eq!(json["event"], "profile.degraded");
        assert_eq!(json["name"], "finance");
        assert_eq!(json["dropped"][0], "big-model");
        assert_eq!(json["shrunk"][0], "qwen35");
        assert_eq!(json["reason"], "vram");

        let loaded = serde_json::to_value(EngineEvent::SlotLoaded {
            id: "embed-gemma".into(),
        })
        .unwrap();
        assert_eq!(loaded["event"], "slot.loaded");
        assert_eq!(loaded["id"], "embed-gemma");
    }

    #[test]
    fn subscribe_replays_then_forwards_live() {
        let bus = EventBus::new();
        bus.emit(EngineEvent::SlotLoaded { id: "a".into() });
        bus.emit(EngineEvent::SlotUnloaded { id: "b".into() });

        let (mut replay, mut rx) = bus.subscribe();
        assert_eq!(replay.len(), 2);
        assert!(replay[0].contains("\"slot.loaded\""));

        bus.emit(EngineEvent::SlotLoaded { id: "c".into() });
        let live = rx.blocking_recv().unwrap();
        assert!(live.contains("\"slot.loaded\""));
        assert!(live.contains("\"id\":\"c\""));
        replay.clear();
    }

    #[test]
    fn replay_buffer_is_capped() {
        let bus = EventBus::new();
        for i in 0..(REPLAY_CAP + 25) {
            bus.emit(EngineEvent::SlotLoaded {
                id: format!("m{i}"),
            });
        }
        let (replay, _) = bus.subscribe();
        assert_eq!(replay.len(), REPLAY_CAP);
        assert!(replay.last().unwrap().contains("m124"));
    }
}
