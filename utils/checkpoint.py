
import os

def load(self, f=None):
    if self.has_checkpoint():
        # override argument with existing checkpoint
        f = self.get_checkpoint_file()
    if not f:
        # no checkpoint could be found
        self.logger.info("No checkpoint found. Initializing model from scratch")
        return {}
    self.logger.info("Loading checkpoint from {}".format(f))
    checkpoint = self._load_file(f)
    self._load_model(checkpoint)
    if "optimizer" in checkpoint and self.optimizer:
        self.logger.info("Loading optimizer from {}".format(f))
        self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
    if "scheduler" in checkpoint and self.scheduler:
        self.logger.info("Loading scheduler from {}".format(f))
        self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

    # return any further checkpoint data
    return checkpoint

def has_checkpoint(self):
    save_file = os.path.join(self.save_dir, "last_checkpoint")
    return os.path.exists(save_file)

def get_checkpoint_file(self):
    save_file = os.path.join(self.save_dir, "last_checkpoint")
    try:
        with open(save_file, "r") as f:
            last_saved = f.read()
    except IOError:
        # if file doesn't exist, maybe because it has just been
        # deleted by a separate process
        last_saved = ""
    return last_saved