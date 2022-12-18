use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::task::TaskId;

const WORKER_THREAD_NAME_BASE: &'static str = concat!(env!("CARGO_PKG_NAME"), " worker");

pub type Job = Box<dyn FnOnce() + Send>;

pub type WorkerId = usize;

struct Worker {
    thread: JoinHandle<()>,
    job_queue: mpsc::SyncSender<(JobInfo, Job)>,
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    available: Vec<WorkerId>,
    finished: mpsc::Receiver<(JobInfo, WorkerId)>,
}

pub struct JobInfo {
    pub caller: TaskId,
    pub job_id: TaskId,
}

impl ThreadPool {
    pub fn new(num_workers: usize) -> Self {
        let (finish_sender, finish_receiver) = mpsc::sync_channel(num_workers);
        ThreadPool {
            available: (0..num_workers).collect(),
            workers: (0..num_workers)
                .map(|id| {
                    let (task_sender, task_receiver) = mpsc::sync_channel::<(JobInfo, Job)>(0);
                    let finish_sender = finish_sender.clone();
                    Worker {
                        thread: std::thread::Builder::new()
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

    pub fn stop(&mut self) {
        let handles = self
            .workers
            .drain(..)
            .map(|w| {
                std::mem::drop(w.job_queue);
                w.thread
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    }

    pub fn finished(&mut self) -> Option<JobInfo> {
        match self.finished.try_recv() {
            Ok((info, worker_id)) => {
                self.available.push(worker_id);
                Some(info)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => panic!("Thread terminated"),
        }
    }

    pub fn worker_available(&self) -> bool {
        !self.available.is_empty()
    }

    pub fn submit(&mut self, info: JobInfo, job: Job) {
        let worker_id = self.available.pop().unwrap();
        self.workers[worker_id].job_queue.send((info, job)).unwrap();
    }
}
