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
        let (task_sender, task_receiver) = mpsc::sync_channel::<(JobInfo, Job)>(0);
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
        mut job_receiver: work_queue::LocalQueue<(JobInfo, Job)>,
        result_sender: mpsc::Sender<JobInfo>,
    ) -> ComputeWorker {
        ComputeWorker {
            _thread: std::thread::Builder::new()
                .name(format!("{} {}", thread_name_prefix, id))
                .spawn(move || {
                    let initial_sleep_amount = std::time::Duration::from_micros(1);
                    let max_sleep_amount = std::time::Duration::from_millis(10);
                    let mut sleep_amount = initial_sleep_amount;
                    loop {
                        let (job_info, task) = loop {
                            if let Some(t) = job_receiver.pop() {
                                break t;
                            }
                            std::thread::sleep(sleep_amount);
                            sleep_amount = (2 * sleep_amount).min(max_sleep_amount);
                        };
                        task();
                        let _ = result_sender.send(job_info);
                        sleep_amount = initial_sleep_amount;
                    }
                })
                .unwrap(),
        }
    }
}

pub struct ComputeThreadPool {
    _workers: Vec<ComputeWorker>,
    job_sender: work_queue::Queue<(JobInfo, Job)>,
}

impl ComputeThreadPool {
    pub fn new(result_sender: mpsc::Sender<JobInfo>, num_workers: usize) -> Self {
        let num_local_items = 16;
        let job_sender = work_queue::Queue::new(num_workers, num_local_items);
        let name = "compute";
        ComputeThreadPool {
            _workers: job_sender
                .local_queues()
                .enumerate()
                .map(|(id, receiver)| ComputeWorker::new(name, id, receiver, result_sender.clone()))
                .collect(),
            job_sender,
        }
    }

    pub fn submit(&mut self, info: JobInfo, job: Job) {
        self.job_sender.push((info, job));
    }
}
