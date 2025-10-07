
TODO: 代码复用和优化


TODO: 可做详细判断，而不是简单跳过，可以分析一下
ZigZagHL 处理上有问题，直接跳过，潜在风险, TODO: 更好的处理方式
        if self.i_low == self.i_high:
            return confirmed_at, self.searching_dir, self.i_high, self.i_low
## TODOs
- reset 未被测试

## Note
使用pyta直接移过来的，只是默认不再为None，而是负整数，confirmed_at = self._default_confirmed_at
