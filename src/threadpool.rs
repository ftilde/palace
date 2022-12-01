use std::cell::Cell;
use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::operator::OperatorId;
use crate::task::{Request, TaskId};

const WORKER_THREAD_NAME_BASE: &'static str = concat!(env!("CARGO_PKG_NAME"), " worker");

pub type Job = Box<dyn FnOnce() + Send>;

pub type WorkerId = usize;

struct Worker {
    _thread: JoinHandle<()>,
    job_queue: mpsc::SyncSender<(TaskId, Job)>,
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    available: Vec<WorkerId>,
    finished: mpsc::Receiver<(TaskId, WorkerId)>,
}

pub fn create_pool(num_workers: usize) -> (ThreadPool, ThreadSpawner) {
    (
        ThreadPool::new(num_workers),
        ThreadSpawner {
            job_counter: Cell::new(0),
        },
    )
}

impl ThreadPool {
    fn new(num_workers: usize) -> Self {
        let (finish_sender, finish_receiver) = mpsc::sync_channel(num_workers);
        ThreadPool {
            available: (0..num_workers).collect(),
            workers: (0..num_workers)
                .map(|id| {
                    let (task_sender, task_receiver) = mpsc::sync_channel::<(TaskId, Job)>(0);
                    let finish_sender = finish_sender.clone();
                    Worker {
                        _thread: std::thread::Builder::new()
                            .name(format!("{} {}", WORKER_THREAD_NAME_BASE, id))
                            .spawn(move || {
                                while let Ok((task_id, task)) = task_receiver.recv() {
                                    task();
                                    let _ = finish_sender.send((task_id, id));
                                }
                            })
                            .unwrap(),
                        job_queue: task_sender,
                    }
                })
                .collect(),
            finished: finish_receiver,
        }
    }

    pub fn finished(&mut self) -> Option<TaskId> {
        match self.finished.try_recv() {
            Ok((task_id, worker_id)) => {
                self.available.push(worker_id);
                Some(task_id)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => panic!("Thread terminated"),
        }
    }

    pub fn worker_available(&self) -> bool {
        !self.available.is_empty()
    }

    pub fn submit(&mut self, task_id: TaskId, job: Job) {
        let worker_id = self.available.pop().unwrap();
        self.workers[worker_id]
            .job_queue
            .send((task_id, job))
            .unwrap();
    }
}

pub struct ThreadSpawner {
    job_counter: Cell<usize>,
}

impl ThreadSpawner {
    pub fn spawn<'req, 'op>(&'req self, f: impl FnOnce() + Send + 'req) -> Request<'req, 'op, ()> {
        let job_num = self.job_counter.get() + 1;
        self.job_counter.set(job_num);

        // TODO: This is a giant hack. possibly revisit the concept of TaskId altogether
        use crate::operator::Operator;
        let id = OperatorId::new::<Self>(&[job_num.id()]);
        let id = id.inner().into();

        let f: Box<dyn FnOnce() + Send + 'req> = Box::new(f);

        // TODO: Ensure the safety of this. I think this is fine as long as we make sure to stop
        // any compute threads before the runtime (and corresponding tasks) are dropped.
        let f = unsafe { std::mem::transmute(f) };

        Request {
            id,
            type_: crate::task::RequestType::ThreadPoolJob(f),
            poll: Box::new(move |_ctx| Some(&())),
            _marker: Default::default(),
        }
    }
}
