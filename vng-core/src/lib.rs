pub mod aabb;
pub mod array;
pub mod chunk_utils;
pub mod coordinate;
pub mod data;
pub mod dim;
pub mod event;
pub mod id;
pub mod mat;
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
pub mod vec;
pub mod vulkan;

// Reexports of dependencies
pub use cgmath;

// TODO look into thiserror/anyhow
pub type Error = Box<(dyn std::error::Error + 'static)>;
