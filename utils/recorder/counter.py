import math


class TrainingCounter:
    def __init__(self, epoch_length, epoch_based=True, *, num_epochs=None, num_total_iters=None) -> None:
        self.num_inner_iters = epoch_length
        self._iter_counter = 0
        self._epoch_counter = 0

        if epoch_based:
            assert num_epochs is not None
            self.num_epochs = num_epochs
            self.num_total_iters = num_epochs * epoch_length
        else:
            assert num_total_iters is not None
            self.num_total_iters = num_total_iters
            self.num_epochs = math.ceil(num_total_iters / epoch_length)

    def set_start_epoch(self, start_epoch):
        self._epoch_counter = start_epoch
        self._iter_counter = start_epoch * self.num_inner_iters

    def set_start_iterations(self, start_iteration):
        self._iter_counter = start_iteration
        self._epoch_counter = start_iteration // self.num_inner_iters

    def every_n_epochs(self, n: int) -> bool:
        return (self._epoch_counter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, n: int) -> bool:
        return (self._iter_counter + 1) % n == 0 if n > 0 else False

    def is_first_epoch(self) -> bool:
        return self._epoch_counter == 0

    def is_last_epoch(self) -> bool:
        return self._epoch_counter == self.num_epochs - 1

    def is_first_inner_iter(self) -> bool:
        return self._iter_counter % self.num_inner_iters == 0

    def is_last_inner_iter(self) -> bool:
        return (self._iter_counter + 1) % self.num_inner_iters == 0

    def is_first_total_iter(self) -> bool:
        return self._iter_counter == 0

    def is_last_total_iter(self) -> bool:
        return self._iter_counter == self.num_total_iters - 1

    def update_iter_counter(self):
        self._iter_counter += 1

    def update_epoch_counter(self):
        self._epoch_counter += 1

    def reset_iter_all_counter(self):
        self._iter_counter = 0
        self._epoch_counter = 0

    @property
    def curr_iter(self):
        return self._iter_counter

    @property
    def next_iter(self):
        return self._iter_counter + 1

    @property
    def curr_epoch(self):
        return self._epoch_counter

    @property
    def curr_percent(self):
        return self._iter_counter / self.num_total_iters
