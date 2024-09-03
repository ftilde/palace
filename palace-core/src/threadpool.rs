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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
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

struct IoWorker {
    _thread: JoinHandle<()>,
    job_queue: mpsc::SyncSender<(JobInfo, Job)>,
}

impl IoWorker {
    fn new(
        thread_name_prefix: &str,
        id: usize,
        finish_sender: mpsc::Sender<WorkerId>,
        result_sender: mpsc::Sender<JobInfo>,
    ) -> IoWorker {
        let (task_sender, task_receiver) = mpsc::sync_channel::<(JobInfo, Job)>(1);
        IoWorker {
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

pub struct IoThreadPool {
    workers: Vec<IoWorker>,
    available: Vec<WorkerId>,
    finished: mpsc::Receiver<WorkerId>,
    finished_sender: mpsc::Sender<WorkerId>,
    result_sender: mpsc::Sender<JobInfo>,
    name: &'static str,
}

impl IoThreadPool {
    pub fn new(result_sender: mpsc::Sender<JobInfo>) -> Self {
        let num_workers = 4;
        let (finished_sender, finish_receiver) = mpsc::channel();
        let name = "io";
        IoThreadPool {
            available: (0..num_workers).collect(),
            workers: (0..num_workers)
                .map(|id| IoWorker::new(name, id, finished_sender.clone(), result_sender.clone()))
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
            self.workers.push(IoWorker::new(
                self.name,
                id,
                self.finished_sender.clone(),
                self.result_sender.clone(),
            ));
        }
    }

    fn collect_finished(&mut self) {
        self.available.extend(self.finished.try_iter());
    }
    fn worker_available(&self) -> bool {
        !self.available.is_empty()
    }

    pub fn submit(&mut self, info: JobInfo, job: Job) {
        self.collect_finished();
        if !self.worker_available() {
            let current_size = self.workers.len();
            self.expand_pool(current_size);
        }

        let worker_id = self.available.pop().unwrap();
        self.workers[worker_id].job_queue.send((info, job)).unwrap();
    }
}

struct ComputeWorker {
    _thread: JoinHandle<()>,
}

impl ComputeWorker {
    fn new(
        thread_name_prefix: &str,
        id: usize,
        job_receiver: spmc::Receiver<(JobInfo, Job)>,
        result_sender: mpsc::Sender<JobInfo>,
    ) -> ComputeWorker {
        ComputeWorker {
            _thread: std::thread::Builder::new()
                .name(format!("{} {}", thread_name_prefix, id))
                .spawn(move || {
                    while let Ok((job_info, task)) = job_receiver.recv() {
                        task();
                        let _ = result_sender.send(job_info);
                    }
                })
                .unwrap(),
        }
    }
}

pub struct ComputeThreadPool {
    _workers: Vec<ComputeWorker>,
    job_sender: spmc::Sender<(JobInfo, Job)>,
}

impl ComputeThreadPool {
    pub fn new(result_sender: mpsc::Sender<JobInfo>, num_workers: usize) -> Self {
        let (job_sender, job_receiver) = spmc::channel();
        let name = "compute";
        ComputeThreadPool {
            _workers: (0..num_workers)
                .into_iter()
                .map(|id| ComputeWorker::new(name, id, job_receiver.clone(), result_sender.clone()))
                .collect(),
            job_sender,
        }
    }

    pub fn submit(&mut self, info: JobInfo, job: Job) {
        self.job_sender.send((info, job)).unwrap();
    }
}
