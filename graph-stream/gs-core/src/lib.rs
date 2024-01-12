use std::{io::Write, path::Path};

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub label: String,
}

#[derive(Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Event {
    AddNode(Node),
    RemoveNode(Node),
    AddEdge(Edge),
    RemoveEdge(Edge),
}

impl Event {
    pub fn inverse(self) -> Event {
        match self {
            Event::AddNode(n) => Event::RemoveNode(n),
            Event::RemoveNode(n) => Event::AddNode(n),
            Event::AddEdge(e) => Event::RemoveEdge(e),
            Event::RemoveEdge(e) => Event::AddEdge(e),
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct EventStream(pub Vec<Event>);

impl EventStream {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn add(&mut self, e: Event) {
        self.0.push(e);
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
        serde_json::to_writer(&mut writer, self).unwrap();
        writer.flush().unwrap();
    }
}
