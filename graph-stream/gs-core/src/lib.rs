use std::time::Instant;
use std::{io::Write, path::Path};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub label: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub label: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Event {
    AddNode(Node),
    RemoveNode(Node),
    AddEdge(Edge),
    RemoveEdge(Edge),
    UpdateEdgeLabel(Edge, String),
}

impl Event {
    pub fn inverse(self) -> Event {
        match self {
            Event::AddNode(n) => Event::RemoveNode(n),
            Event::RemoveNode(n) => Event::AddNode(n),
            Event::AddEdge(e) => Event::RemoveEdge(e),
            Event::RemoveEdge(e) => Event::AddEdge(e),
            Event::UpdateEdgeLabel(e, label) => Event::UpdateEdgeLabel(
                Edge {
                    from: e.from,
                    to: e.to,
                    label,
                },
                e.label,
            ),
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Timestamp(u64);
impl Timestamp {
    pub fn ms(&self) -> u64 {
        self.0
    }
    pub fn from_ms(ms: u64) -> Self {
        Timestamp(ms)
    }
}

pub struct EventStreamBuilder {
    pub stream: EventStream,
    begin_ts: Instant,
}

impl Default for EventStreamBuilder {
    fn default() -> Self {
        Self {
            stream: EventStream(Vec::new()),
            begin_ts: Instant::now(),
        }
    }
}
impl EventStreamBuilder {
    pub fn add(&mut self, e: Event) {
        self.stream
            .0
            .push((Timestamp(self.begin_ts.elapsed().as_millis() as _), e));
    }
}

#[derive(Serialize, Deserialize)]
pub struct EventStream(pub Vec<(Timestamp, Event)>);

impl EventStream {
    pub fn begin_ts(&self) -> Timestamp {
        self.0.first().map(|v| v.0).unwrap()
    }
    pub fn end_ts(&self) -> Timestamp {
        self.0.last().map(|v| v.0).unwrap()
    }

    pub fn load(path: &Path) -> Self {
        let reader = std::fs::OpenOptions::new().read(true).open(path).unwrap();
        let mut reader = std::io::BufReader::new(reader);
        serde_json::from_reader(&mut reader).unwrap()
    }

    pub fn save(&self, path: &Path) {
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)
            .unwrap();
        let mut writer = std::io::BufWriter::new(writer);
        serde_json::to_writer_pretty(&mut writer, self).unwrap();
        writer.flush().unwrap();
    }
}
