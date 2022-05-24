__all__ = [
    "get_same_padding",
    "val2list",
    "make_divisible",
]


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count