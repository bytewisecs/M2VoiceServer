"""各数据入口共用的已知无效采集会话。

20260411_171730 沿用 mDataloader/mDataset.py 原有排除规则。
这些会话仅从划分和训练中排除，不删除原始文件。
"""

INVALID_GROUPS = frozenset({"20260411_171730"})
