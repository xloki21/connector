from connector.dataloader import _MultiWorkerIter
from connector.tools.sampler import BatchSampler


def creating_task_for_pool(sampler, batch_size, worker_pool, worker_fn, prefetch, last_batch='discard'):

    batch_sampler = BatchSampler(sampler, batch_size=batch_size, last_batch=last_batch)

    stat_iter = _MultiWorkerIter(worker_pool=worker_pool,
                                 batch_sampler=batch_sampler,
                                 worker_fn=worker_fn,
                                 prefetch=prefetch)

    return stat_iter
