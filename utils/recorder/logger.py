from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, tb_root):
        self.tb_root = tb_root
        self.tb = None

    def write_to_tb(self, name, data, curr_iter):
        assert self.tb_root is not None

        if self.tb is None:
            self.tb = SummaryWriter(self.tb_root)

        if not isinstance(data, (tuple, list)):
            self.tb.add_scalar(f"data/{name}", data, curr_iter)
        else:
            for idx, data_item in enumerate(data):
                self.tb.add_scalar(f"data/{name}_{idx}", data_item, curr_iter)

    def close_tb(self):
        if self.tb is not None:
            self.tb.close()
