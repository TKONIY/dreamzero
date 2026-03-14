# print_utils.py - 打印与日志工具

## 文件概述

`print_utils.py` 提供了一套丰富的打印、格式化和输出控制工具，涵盖以下功能领域：

1. **数值格式化**：可读的数字缩写（如 `1.5M`）、科学计数法
2. **字符串工具**：f-string 模拟、格式键提取、时间戳生成
3. **对象表示**：美化的 `__repr__` 生成器
4. **调试打印**：支持张量形状摘要的调试打印器
5. **输出重定向**：将 stdout/stderr 重定向到文件、字符串或 `/dev/null`
6. **日志过滤**：基于模式匹配的 logging 过滤器
7. **装饰器**：`@watch` 实现类似 Unix `watch` 命令的周期性执行

## 关键代码解析

### 1. 可读数字格式化

```python
def to_readable_count_str(value: int, precision: int = 2) -> str:
```

将大数字转换为人类可读格式：`1500000` -> `"1.50 M"`。使用 K/M/B/T 后缀，最多到万亿级别。主要用于显示模型参数量。

### 2. DebugPrinter 类

```python
class DebugPrinter:
    def __init__(self, enabled, tensor_summary="shape"):
```

一个可开关的调试打印器，特别针对深度学习场景优化：
- `shape` 模式：张量只打印形状，如 `[3, 224, 224]`
- `shape+dtype` 模式：还打印数据类型和设备，如 `float32[3, 224, 224]|cuda:0`
- `none` 模式：打印完整张量

使用示例：
```python
dprint = DebugPrinter(enabled=True)
dprint(my_tensor)  # 只打印 [3, 224, 224]
```

### 3. 输出重定向体系

模块实现了一个层次化的输出重定向系统：

```
PrintRedirection（基类，重定向 stdout/stderr）
├── PrintToFile（重定向到文件）
│   └── PrintSuppress（重定向到 /dev/null）
└── PrintString（重定向到字符串缓冲区）
```

`PrintRedirection` 是上下文管理器，在 `__enter__` 时替换 `sys.stdout`/`sys.stderr`，在 `__exit__` 时恢复。`PrintString` 可以用于捕获函数的输出并作为字符串返回。

### 4. `@watch` 装饰器

```python
@meta_decorator
def watch(func, seconds=5, max_times=0, keep_returns=False):
```

模拟 Unix `watch -n` 命令，周期性执行函数并使用 `blessings.Terminal` 实现全屏刷新效果。适用于监控脚本。

### 5. 日志过滤器

```python
class ExcludeLoggingFilter(logging.Filter):
class ReplaceStringLoggingFilter(logging.Filter):
```

基于通配符模式匹配的 logging 过滤器。`ExcludeLoggingFilter` 可以屏蔽匹配的日志消息，`ReplaceStringLoggingFilter` 可以替换匹配的日志内容。

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `to_readable_count_str(value)` | 将整数转为可读字符串（如 `"1.50 M"`） |
| `to_scientific_str(value)` | 将浮点数转为科学计数法字符串 |
| `print_str(*args)` | 与 `print()` 相同签名，但返回字符串 |
| `fstring(fmt_str, **kwargs)` | 模拟 f-string，运行时求值 |
| `get_format_keys(fmt_str)` | 提取格式字符串中的字段名 |
| `get_timestamp()` | 获取当前时间戳字符串 |
| `pretty_repr_str(obj, **kwargs)` | 生成美化的 `__repr__` 字符串 |
| `pprint_(*objs)` | 使用 pprint 格式化打印 |
| `get_exception_info()` | 获取当前异常的详细信息字典 |
| `DebugPrinter` 类 | 可开关的调试打印器，支持张量摘要 |
| `watch` 装饰器 | 周期性执行函数（类似 Unix watch） |
| `PrintRedirection` 类 | stdout/stderr 重定向上下文管理器 |
| `PrintToFile` 类 | 将输出重定向到文件 |
| `PrintSuppress` 函数 | 抑制输出（重定向到 /dev/null） |
| `PrintString` 类 | 将输出捕获为字符串 |
| `ExcludeLoggingFilter` 类 | 基于模式匹配排除日志消息 |
| `ReplaceStringLoggingFilter` 类 | 基于模式匹配替换日志消息 |
| `logging_exclude_pattern(...)` | 便捷函数，为 logger 添加排除过滤器 |
| `logging_replace_string(...)` | 便捷函数，为 logger 添加替换过滤器 |

## 与其他模块的关系

- **依赖** `functional_utils.py`：使用 `meta_decorator` 装饰 `@watch`
- **依赖** `misc_utils.py`：使用 `match_patterns` 实现日志过滤的模式匹配
- **被** `config_utils.py` 依赖：`to_scientific_str` 用于 OmegaConf 的 `scientific` 解析器
- **被** `torch_utils.py` 依赖：`to_readable_count_str` 用于显示模型参数量

## 总结

`print_utils.py` 是一个面向深度学习开发的打印和日志工具集。`DebugPrinter` 和张量摘要功能显著降低了调试神经网络时的信息过载；输出重定向系统提供了灵活的输出控制能力；日志过滤器则帮助在复杂训练流程中管理大量日志输出。`to_readable_count_str` 和 `to_scientific_str` 在训练报告和配置命名中有广泛应用。
