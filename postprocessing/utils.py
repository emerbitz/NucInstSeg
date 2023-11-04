import functools

from torch.testing import assert_close

assert_equal = functools.partial(assert_close, atol=0, rtol=0, check_device=False)
