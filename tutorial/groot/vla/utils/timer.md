# `timer.py` — 可嵌套的上下文计时器

## 文件概述

本文件实现了一个轻量级的**上下文管理器计时工具** `ContextTimer`，用于在训练循环中精确测量各阶段（如前向传播、反向传播、数据加载等）的耗时，并将结果通过 trainer 的日志接口自动记录。其设计支持嵌套使用，可以同时测量外层和内层代码块的执行时间而互不干扰。

## 关键代码解析

### `ContextTimer` 类

```python
class ContextTimer:
    def __init__(self, trainer):
        self.last_key = None
        self.trainer = trainer
        self.start_times = {}
        self.key_stack = []
```

**核心数据结构**：
- `trainer`：持有 trainer 实例的引用，用于调用 `trainer.log()` 记录耗时数据。
- `start_times`：字典，以标签名为键、起始时间戳为值，支持多个计时器并行存在。
- `key_stack`：栈结构，用于处理嵌套调用时正确匹配 `__enter__` 和 `__exit__` 的标签。
- `last_key`：临时存储最近一次 `with_label()` 设置的标签名，作为 `__enter__` 的输入。

### `with_label` 方法

```python
def with_label(self, key):
    self.last_key = key
    return self
```

采用**链式调用**模式，设置当前计时块的标签名并返回 `self`，使得可以直接在 `with` 语句中使用：

```python
with my_timer.with_label("forward_pass"):
    # 被计时的代码
```

`with_label` 返回 `self`（即 `ContextTimer` 实例本身），因此 Python 会接着调用该实例的 `__enter__` 方法进入上下文。

### `__enter__` 和 `__exit__`

```python
def __enter__(self):
    self.key_stack.append(self.last_key)
    self.start_times[self.last_key] = time.time()
    return self

def __exit__(self, exc_type, exc_value, traceback):
    key = self.key_stack.pop()
    diff = time.time() - self.start_times[key]
    self.trainer.log({f"{key}_time": diff})
```

- **进入上下文时**：将标签压入栈中，同时记录当前时间戳。
- **退出上下文时**：从栈中弹出标签，计算耗时差值，以 `{标签}_time` 为键调用 `trainer.log()` 记录。

栈结构保证了嵌套场景的正确性。例如：

```python
with my_timer.with_label("outer"):       # 压入 "outer"
    with my_timer.with_label("inner"):    # 压入 "inner"
        pass                              # 弹出 "inner"，记录 inner_time
    # 继续 outer 的计时
# 弹出 "outer"，记录 outer_time（包含 inner 的时间）
```

### 测试代码（`__main__` 块）

文件末尾包含一段自测代码，使用 `MockTrainer` 模拟 trainer 的日志接口，演示了嵌套计时的典型用法：外层 `"outer"` 包含 1 秒等待 + 两个内层 `"inner"` 块（分别 2 秒和 1 秒）。预期输出为 `inner_time` 约 2 秒和 1 秒、`outer_time` 约 4 秒。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `ContextTimer` | 类 | 可嵌套的上下文管理器计时工具 |
| `ContextTimer.__init__` | 方法 | 初始化，绑定 trainer 实例，创建计时栈和字典 |
| `ContextTimer.with_label` | 方法 | 设置当前计时块的标签名，返回 self 以支持 `with` 语法 |
| `ContextTimer.__enter__` | 方法 | 压栈并记录起始时间 |
| `ContextTimer.__exit__` | 方法 | 弹栈、计算耗时、调用 `trainer.log()` 记录 |

## 与其他模块的关系

- **`groot/vla/experiment/base.py`**：在基类 `BaseExperiment` 的初始化阶段（第 329 行）创建 `self.timer = ContextTimer(self)`，随后在训练循环的各阶段使用 `self.timer.with_label(...)` 来测量耗时。trainer 本身实现了 `log()` 方法，因此可以直接作为 `ContextTimer` 的 `trainer` 参数。
- **日志系统**：`trainer.log()` 通常会将数据发送到 WandB 或其他日志后端，使得计时数据可以在训练监控面板中可视化。

## 总结

`timer.py` 提供了一个精巧的计时工具 `ContextTimer`，通过 Python 上下文管理器协议和栈结构实现了可嵌套、自动记录的计时功能。其 `with_label` 链式调用设计使得使用语法简洁自然，而与 trainer 日志系统的集成则让计时数据可以直接用于训练过程的性能分析和瓶颈定位。
