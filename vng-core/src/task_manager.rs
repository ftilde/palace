use std::{cell::Cell, collections::BTreeMap, time::Duration};

use crate::{
    task::{Request, Task, ThreadPoolJob},
    task_graph::TaskId,
    threadpool::{ComputeThreadPool, IoThreadPool, JobId, JobInfo, JobType},
};

struct TaskData<'a> {
    task: Task<'a>,
    waiting_on_threads: usize,
}

impl TaskData<'_> {
    fn pollable(&self) -> bool {
        self.waiting_on_threads == 0
    }
}

pub struct TaskManager<'a> {
    managed_tasks: BTreeMap<TaskId, TaskData<'a>>,
    compute_thread_pool: &'a mut ComputeThreadPool,
    io_thread_pool: &'a mut IoThreadPool,
    async_result_receiver: &'a mut std::sync::mpsc::Receiver<JobInfo>,
}

impl Drop for TaskManager<'_> {
    fn drop(&mut self) {
        while self
            .managed_tasks
            .values()
            .any(|d| d.waiting_on_threads > 0)
        {
            let _ = self.wait_for_jobs(Duration::from_secs(u64::max_value()));
        }
        // First wait for all relevant jobs on the thread pool finish THEN drop tasks
    }
}

#[derive(Debug)]
pub enum Error {
    NoSuchTask,
    WaitingOnThreads,
}

impl<'a> TaskManager<'a> {
    pub fn new(
        compute_thread_pool: &'a mut ComputeThreadPool,
        io_thread_pool: &'a mut IoThreadPool,
        async_result_receiver: &'a mut std::sync::mpsc::Receiver<JobInfo>,
    ) -> Self {
        Self {
            managed_tasks: BTreeMap::new(),
            compute_thread_pool,
            io_thread_pool,
            async_result_receiver,
        }
    }

    pub fn add_task(&mut self, task_id: TaskId, task: Task<'a>) -> &mut Task<'a> {
        self.managed_tasks.insert(
            task_id,
            TaskData {
                task,
                waiting_on_threads: 0,
            },
        );
        self.managed_tasks
            .get_mut(&task_id)
            .map(|v| &mut v.task)
            .unwrap()
    }

    pub fn has_task(&self, task_id: TaskId) -> bool {
        self.managed_tasks.contains_key(&task_id)
    }

    pub fn get_task(&mut self, task_id: TaskId) -> Result<&mut Task<'a>, Error> {
        let task_data = self
            .managed_tasks
            .get_mut(&task_id)
            .ok_or(Error::NoSuchTask)?;
        Ok(&mut task_data.task)
    }

    pub fn remove_task(&mut self, task_id: TaskId) -> Result<(), Error> {
        let task_data = self.managed_tasks.get(&task_id).ok_or(Error::NoSuchTask)?;
        if task_data.pollable() {
            self.managed_tasks.remove(&task_id);
            Ok(())
        } else {
            Err(Error::WaitingOnThreads)
        }
    }

    pub fn wait_for_jobs(&mut self, timeout: Duration) -> Vec<JobId> {
        let mut result = Vec::new();
        let info = match self.async_result_receiver.recv_timeout(timeout) {
            Ok(info) => info,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return result,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("Job result pipe disconnected")
            }
        };

        let mut job_done = |info: JobInfo| {
            self.managed_tasks
                .get_mut(&info.caller)
                .unwrap()
                .waiting_on_threads -= 1;
            info.job_id
        };

        result.push(job_done(info));

        // Try to empty the queue after waiting for the first event to arrive
        for info in self.async_result_receiver.try_iter() {
            result.push(job_done(info));
        }

        result
    }

    pub fn spawn_job(&mut self, job: ThreadPoolJob, type_: JobType) -> Result<(), Error> {
        let task_data = self
            .managed_tasks
            .get_mut(&job.waiting_id)
            .ok_or(Error::NoSuchTask)?;
        task_data.waiting_on_threads += 1;

        let info = crate::threadpool::JobInfo {
            caller: job.waiting_id,
            job_id: job.id,
        };
        match type_ {
            JobType::Compute => self.compute_thread_pool.submit(info, job.job),
            JobType::Io => self.io_thread_pool.submit(info, job.job),
        };
        Ok(())
    }
}

pub struct ThreadSpawner {
    job_counter: Cell<usize>,
}

impl ThreadSpawner {
    pub fn new() -> Self {
        Self {
            job_counter: Cell::new(0),
        }
    }

    #[must_use]
    pub fn spawn<'req, 'irrelevant, R: Send + 'req>(
        &'req self,
        type_: JobType,
        caller: TaskId,
        f: impl FnOnce() -> R + Send + 'req,
    ) -> Request<'req, 'irrelevant, R> {
        // Note that the lifetime 'irrelevant is (unsurprisingly) irrelevant since it is only used
        // in the data variant of Request/RequestType

        let job_num = self.job_counter.get() + 1;

        // TODO Probably remove at some point, but for now to ease debugging: Compute jobs get an
        // even id, io jobs an odd.
        let job_num = match (type_, job_num % 2) {
            (JobType::Compute, 0) => job_num,
            (JobType::Compute, _) => job_num + 1,
            (JobType::Io, 1) => job_num,
            (JobType::Io, _) => job_num + 1,
        };
        self.job_counter.set(job_num);

        let id = JobId::new(job_num);

        let (result_sender, result_receiver) = oneshot::channel();

        let f = move || {
            let res = f();
            result_sender.send(res).unwrap();
        };

        let f: Box<dyn FnOnce() + Send + 'req> = Box::new(f);

        // Safety: We extend the lifetime of the job to 'static through the transmute.
        // For this to be safe we need to ensure that the job actually does not run longer than the
        // specified lifetime. This means that we need to control how long the lifetime (within
        // a task!) is:
        //  1. Tasks cannot be removed from the TaskManager as long as jobs are running for it.
        //  2. Tasks cannot return waiting on the job before it is done (see
        //     result_receiver/result_sender above and below)
        //  3. Tasks are not dropped as long as they are waiting on jobs
        let f = unsafe { std::mem::transmute(f) };

        let job = ThreadPoolJob {
            id,
            waiting_id: caller,
            job: f,
        };
        Request {
            type_: crate::task::RequestType::ThreadPoolJob(job, type_),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || match result_receiver.try_recv() {
                    Ok(res) => Some(res),
                    Err(oneshot::TryRecvError::Empty) => None,
                    Err(oneshot::TryRecvError::Disconnected) => {
                        panic!("Either polled twice or the compute thread was interrupted")
                    }
                })
            }),
            _marker: Default::default(),
        }
    }
}
