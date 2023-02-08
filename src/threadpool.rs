use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::task_graph::TaskId;

pub type Job = Box<dyn FnOnce() + Send>;

pub type WorkerId = usize;

#[derive(Copy, Clone)]
pub enum JobType {
    Compute,
    Io,
}

struct Worker {
    _thread: JoinHandle<()>,
    job_queue: mpsc::SyncSender<(JobInfo, Job)>,
}

impl Worker {
    fn new(
        thread_name_prefix: &str,
        id: usize,
        finish_sender: mpsc::Sender<WorkerId>,
        result_sender: mpsc::Sender<JobInfo>,
    ) -> Worker {
        let (task_sender, task_receiver) = mpsc::sync_channel::<(JobInfo, Job)>(0);
        Worker {
            _thread: std::thread::Builder::new()
                .name(format!("{} {}", thread_name_prefix, id))
                .spawn(move || {
                    while let Ok((job_info, task)) = task_receiver.recv() {
                        task();
                        let _ = finish_sender.send(id);
                        let _ = result_sender.send(job_info);
                    }
                })
                .unwrap(),
            job_queue: task_sender,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct JobId(usize);

impl JobId {
    pub fn new(id: usize) -> Self {
        JobId(id)
    }
}

pub struct JobInfo {
    pub caller: TaskId,
    pub job_id: JobId,
}

struct ThreadPool {
    workers: Vec<Worker>,
    available: Vec<WorkerId>,
    finished: mpsc::Receiver<WorkerId>,
    finished_sender: mpsc::Sender<WorkerId>,
    result_sender: mpsc::Sender<JobInfo>,
    name: &'static str,
}

impl ThreadPool {
    fn new(name: &'static str, result_sender: mpsc::Sender<JobInfo>, num_workers: usize) -> Self {
        let (finished_sender, finish_receiver) = mpsc::channel();
        ThreadPool {
            available: (0..num_workers).collect(),
            workers: (0..num_workers)
                .map(|id| Worker::new(name, id, finished_sender.clone(), result_sender.clone()))
                .collect(),
            finished: finish_receiver,
            finished_sender,
            result_sender,
            name,
        }
    }

    fn expand_pool(&mut self, num_workers: usize) {
        let current_size = self.workers.len();
        for id in current_size..current_size + num_workers {
            self.available.push(id);
            self.workers.push(Worker::new(
                self.name,
                id,
                self.finished_sender.clone(),
                self.result_sender.clone(),
            ));
        }
    }

    fn wait_idle(&mut self) {
        while self.available.len() != self.workers.len() {
            let worker_id = self.finished.recv().unwrap();
            self.available.push(worker_id);
        }
    }

    fn collect_finished(&mut self) {
        self.available.extend(self.finished.try_iter());
    }
    fn worker_available(&self) -> bool {
        !self.available.is_empty()
    }

    fn submit(&mut self, info: JobInfo, job: Job) {
        let worker_id = self.available.pop().unwrap();
        self.workers[worker_id].job_queue.send((info, job)).unwrap();
    }
}

pub struct ComputeThreadPool(ThreadPool);

impl ComputeThreadPool {
    pub fn new(result_sender: mpsc::Sender<JobInfo>, num_compute_workers: usize) -> Self {
        Self(ThreadPool::new(
            "compute",
            result_sender,
            num_compute_workers,
        ))
    }

    pub fn wait_idle(&mut self) {
        self.0.wait_idle()
    }

    pub fn collect_finished(&mut self) {
        self.0.collect_finished();
    }
    pub fn worker_available(&self) -> bool {
        self.0.worker_available()
    }

    pub fn submit(&mut self, info: JobInfo, job: Job) {
        self.0.submit(info, job)
    }
}

pub struct IoThreadPool(ThreadPool);

impl IoThreadPool {
    pub fn new(result_sender: mpsc::Sender<JobInfo>) -> Self {
        Self(ThreadPool::new("io", result_sender, 4))
    }

    pub fn wait_idle(&mut self) {
        self.0.wait_idle()
    }

    pub fn submit(&mut self, info: JobInfo, job: Job) {
        self.0.collect_finished();
        if !self.0.worker_available() {
            let current_size = self.0.workers.len();
            self.0.expand_pool(current_size);
        }
        self.0.submit(info, job)
    }
}
