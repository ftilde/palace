pub mod array;
pub mod data;
pub mod event;
pub mod id;
pub mod operator;
pub mod operators;
pub mod runtime;
pub mod storage;
pub mod task;
pub mod task_graph;
pub mod task_manager;
#[cfg(test)]
pub mod test_util;
pub mod threadpool;
pub mod util;
pub mod vulkan;

// TODO look into thiserror/anyhow
pub type Error = Box<(dyn std::error::Error + 'static)>;
