use std::sync::mpsc;
use std::thread::JoinHandle;
use std::{cell::Cell, task::Poll};

use crate::{operator::OperatorId, task::Request};

const WORKER_THREAD_NAME_BASE: &'static str = concat!(env!("CARGO_PKG_NAME"), " worker");

type Job = Box<dyn FnOnce() + Send>;

struct Worker {
    thread: JoinHandle<()>,
    id: usize,
    queue: mpsc::SyncSender<Job>,
}

pub struct ThreadPool<'tasks> {
    lifetime: std::marker::PhantomData<&'tasks ()>,
    workers: Vec<Worker>,
    job_counter: Cell<usize>,
}

impl<'tasks> ThreadPool<'tasks> {
    pub fn new(num_workers: usize) -> Self {
        ThreadPool {
            lifetime: Default::default(),
            workers: (0..num_workers)
                .map(|id| {
                    let (task_sender, task_receiver) = mpsc::sync_channel::<Job>(0);
                    Worker {
                        id,
                        thread: std::thread::Builder::new()
                            .name(format!("{} {}", WORKER_THREAD_NAME_BASE, id))
                            .spawn(move || {
                                while let Ok(task) = task_receiver.recv() {
                                    task()
                                }
                            })
                            .unwrap(),
                        queue: task_sender,
                    }
                })
                .collect(),
            job_counter: Cell::new(0),
        }
    }

    pub fn spawn(&self, _f: impl FnOnce() + Send + 'tasks) -> Request<'tasks, ()> {
        let job_num = self.job_counter.get() + 1;
        self.job_counter.set(job_num);

        // TODO: This is a giant hack. possibly revisit the concept of TaskId altogether
        use crate::operator::Operator;
        let id = OperatorId::new::<Self>(&[job_num.id()]);
        let id = id.inner().into();
        Request {
            id,
            compute: Box::new(move |_ctx| std::future::poll_fn(|_| Poll::Ready(Ok(()))).into()),
            poll: Box::new(move |_ctx| Some(&())),
        }
    }
}
