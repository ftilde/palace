use std::{cell::Cell, collections::BTreeMap};

use crate::{
    operator::OperatorId,
    task::{Request, Task, TaskId, ThreadPoolJob},
    threadpool::ThreadPool,
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
    thread_pool: ThreadPool,
}

impl Drop for TaskManager<'_> {
    fn drop(&mut self) {
        self.thread_pool.stop();
        // First stop all workers of the thread pool
        // THEN drop tasks
    }
}

#[derive(Debug)]
pub enum Error {
    NoSuchTask,
    WaitingOnThreads,
}

pub fn create_task_manager<'a>(num_workers: usize) -> (TaskManager<'a>, ThreadSpawner) {
    (
        TaskManager::new(num_workers),
        ThreadSpawner {
            job_counter: Cell::new(0),
        },
    )
}

impl<'a> TaskManager<'a> {
    fn new(num_workers: usize) -> Self {
        Self {
            managed_tasks: BTreeMap::new(),
            thread_pool: ThreadPool::new(num_workers),
        }
    }

    pub fn add_task(&mut self, task_id: TaskId, task: Task<'a>) {
        self.managed_tasks.insert(
            task_id,
            TaskData {
                task,
                waiting_on_threads: 0,
            },
        );
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

    pub fn job_finished(&mut self) -> Option<TaskId> {
        match self.thread_pool.finished() {
            Some(info) => {
                self.managed_tasks
                    .get_mut(&info.caller)
                    .unwrap()
                    .waiting_on_threads -= 1;
                Some(info.job_id)
            }
            None => None,
        }
    }

    pub fn pool_worker_available(&self) -> bool {
        self.thread_pool.worker_available()
    }

    pub fn spawn_job<'job>(&mut self, job_id: TaskId, job: ThreadPoolJob) -> Result<(), Error>
    where
        'a: 'job,
    {
        let task_data = self
            .managed_tasks
            .get_mut(&job.waiting_id)
            .ok_or(Error::NoSuchTask)?;
        task_data.waiting_on_threads += 1;

        self.thread_pool.submit(
            crate::threadpool::JobInfo {
                caller: job.waiting_id,
                job_id,
            },
            job.job,
        );
        Ok(())
    }
}

pub struct ThreadSpawner {
    job_counter: Cell<usize>,
}

impl ThreadSpawner {
    pub fn spawn<'req, 'op>(
        &'req self,
        caller: TaskId,
        f: impl FnOnce() + Send + 'req,
    ) -> Request<'req, 'op, ()> {
        let job_num = self.job_counter.get() + 1;
        self.job_counter.set(job_num);

        // TODO: This is a giant hack. possibly revisit the concept of TaskId altogether
        use crate::operator::Operator;
        let id = OperatorId::new::<Self>(&[job_num.id()]);
        let id = id.inner().into();

        let (drop_notifier, drop_checker) = std::sync::mpsc::sync_channel::<()>(0);
        let f = move || {
            f();
            std::mem::drop(drop_notifier);
        };

        let f: Box<dyn FnOnce() + Send + 'req> = Box::new(f);

        // Safety: We extend the lifetime of the job to 'static through the transmute.
        // For this to be safe we need to ensure that the job actually does not run longer than the
        // specified lifetime. This means that we need to control how long the lifetime (within
        // a task!) is:
        //  1. Tasks cannot be removed from the TaskManager as long as jobs are running for it.
        //  2. Tasks cannot return waiting on the job before it is done (see
        //     drop_notifier/drop_checker above and below)
        //  3. Tasks are not dropped before the threadpool is stopped
        let f = unsafe { std::mem::transmute(f) };

        let job = ThreadPoolJob {
            waiting_id: caller,
            job: f,
        };
        Request {
            id,
            type_: crate::task::RequestType::ThreadPoolJob(job),
            poll: Box::new(move |_ctx| match drop_checker.try_recv() {
                Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(()),
                _ => None,
            }),
            _marker: Default::default(),
        }
    }
}
