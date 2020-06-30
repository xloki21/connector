import multiprocessing
import connector.tools.sampler as _sampler
from multiprocessing.pool import ThreadPool


def _thread_worker_fn(samples, batch_fn, dataset):
    return batch_fn([dataset.iloc[i] for i in samples])


class _MultiWorkerIter:
    def __init__(self, worker_pool, batch_sampler, worker_fn, prefetch=0, timeout=120):

        self._worker_pool = worker_pool
        self._batch_sampler = batch_sampler
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._worker_fn = worker_fn
        self._timeout = timeout

        for _ in range(prefetch):
            self._push_next()

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)

        if r is None:
            return

        async_ret = self._worker_pool.apply_async(self._worker_fn, [r])

        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def __next__(self):
        self._push_next()

        if self._rcvd_idx == self._sent_idx:
            raise StopIteration

        ret = self._data_buffer.pop(self._rcvd_idx)
        try:
            batch = ret.get(self._timeout)

            self._rcvd_idx += 1
            return batch

        except multiprocessing.context.TimeoutError:
            msg = "Поток вышел из таймаута. Возможно нужно увеличить из-за долгой работы функции"
            print(msg)
            raise

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._batch_sampler)


class DataLoader:
    """Loads data from a dataset and returns mini-batches of data.
    Parameters
    ----------
    connector : DataFrame
        Source dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.
        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
    timeout : int, default is 120
        The timeout in seconds for each worker to fetch a batch data. Only modify this number
        unless you are experiencing timeout and you know it's due to slow data loading.
        Sometimes full `shared_memory` will cause all workers to hang and causes timeout. In these
        cases please reduce `num_workers` or increase system `shared_memory` size instead.
    """

    def __init__(self, connector, batch_size, shuffle=False, last_batch=None, num_workers=0, timeout=120):

        self._connector = connector
        self._dataset = self._connector.df
        self._timeout = timeout

        assert timeout > 0, "timeout must be positive, given {}".format(timeout)

        if shuffle:
            sampler = _sampler.RandomSampler(len(self._dataset))
        else:
            sampler = _sampler.SequentialSampler(len(self._dataset))

        batch_sampler = _sampler.BatchSampler(
            sampler, batch_size, last_batch if last_batch else 'keep')

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0
        self._worker_pool = None

        # The number of prefetching batches only works if `num_workers` > 0.
        self._prefetch = 2 * self._num_workers

        if self._num_workers > 0:
            self._worker_pool = ThreadPool(self._num_workers)

    def __iter__(self):
        if self._num_workers == 0:
            def same_process_iter():
                for batch in self._batch_sampler:
                    ret = self._connector.collater_fn(batch)
                    yield ret

            return same_process_iter()

        return _MultiWorkerIter(self._worker_pool, self._batch_sampler,
                                worker_fn=self._connector.collater_fn,
                                prefetch=self._prefetch,
                                # dataset=self._dataset,
                                timeout=self._timeout)

    def __len__(self):
        return len(self._batch_sampler)
