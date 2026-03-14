# misc_utils.py - 杂项工具函数

## 文件概述

`misc_utils.py` 汇集了各种通用工具函数和类，涵盖以下功能领域：

1. **环境变量管理**：批量设置/删除环境变量
2. **模式匹配与过滤**：基于通配符或回调函数的字符串匹配
3. **嵌套属性访问**：支持 `"a.b.c"` 格式的嵌套字典/对象访问
4. **事件触发控制**：周期性触发、单次触发、N 次触发
5. **编码与哈希**：Base64 编码、安全哈希

## 关键代码解析

### 1. 模式匹配系统

```python
def match_patterns(item, include=None, exclude=None, *, precedence="exclude"):
```

这是一个灵活的字符串匹配函数，支持三种匹配方式：
- 通配符字符串（如 `"*.py"`）通过 `fnmatch` 匹配
- 可调用对象（lambda 或函数）直接调用判断
- 以上两种可以混合使用

`include` 和 `exclude` 参数控制包含/排除逻辑，`precedence` 参数决定冲突时谁优先。

`filter_patterns` 是其批量版本，对列表中的每个元素应用匹配过滤，还支持 `ordering="include"` 按包含模式的顺序排列结果。

### 2. 嵌套访问工具

```python
def getitem_nested(cfg, key: str):    # cfg["a"]["b"]["c"]
def setitem_nested(cfg, key: str, value):
def getattr_nested(obj, key: str):    # obj.a.b.c
def setattr_nested(obj, key: str, value):
```

这四个函数将点分隔的键字符串（如 `"model.encoder.hidden_size"`）拆解为逐层访问。适用于动态配置覆盖和模型参数查询。

### 3. 事件触发控制

**`PeriodicEvent`**：周期性触发器
```python
event = PeriodicEvent(period=100)
for step in range(1000):
    if event(new_value=step):  # 每 100 步触发一次
        save_checkpoint()
```
值必须单调递增，支持 `new_value` 和 `increment` 两种更新方式。

**`Once`**：单次触发器
```python
once = Once()
for x in data:
    if once():  # 只在第一次调用时返回 True
        print("First item:", x)
```

**`global_once(name)`** 和 **`global_n_times(name, n)`**：全局级别的触发控制，通过名称字符串标识，跨函数/模块共享状态。适用于"这条警告只打印一次"的场景。

**`Every`**：每 N 次触发
```python
every = Every(n=10)
```

### 4. 环境变量管理

```python
def set_os_envs(envs: Dict[str, Any] = None):
```

批量设置环境变量。特殊值 `None` 或 `"__delete__"` 表示删除该环境变量。这与 `config_utils.py` 中的 `_DELETE_ARG` 语义一致。

### 5. 编码与哈希

```python
def encode_base64(obj) -> str:   # pickle -> base64 编码
def decode_base64(s: str):       # base64 -> pickle 解码
def safe_hash(input_tuple):      # SHA-256 哈希，返回 128 位整数
```

`safe_hash` 特别适合需要从任意 Python 元组生成确定性种子的场景。

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `set_os_envs(envs)` | 批量设置/删除环境变量 |
| `argmax(L)` | 返回列表最大值的索引 |
| `match_patterns(item, include, exclude)` | 灵活的字符串模式匹配 |
| `filter_patterns(items, include, exclude)` | 对列表进行模式过滤 |
| `getitem_nested(cfg, key)` | 点分隔键的嵌套字典访问 |
| `setitem_nested(cfg, key, value)` | 点分隔键的嵌套字典赋值 |
| `getattr_nested(obj, key)` | 点分隔键的嵌套属性访问 |
| `setattr_nested(obj, key, value)` | 点分隔键的嵌套属性赋值 |
| `PeriodicEvent` 类 | 周期性事件触发器 |
| `Once` 类 | 单次触发器 |
| `global_once(name)` | 全局单次触发（按名称） |
| `global_n_times(name, n)` | 全局 N 次触发 |
| `Every` 类 | 每 N 次触发 |
| `encode_base64(obj)` / `decode_base64(s)` | Base64 编码/解码 |
| `safe_hash(input_tuple)` | 确定性哈希（SHA-256，128 位） |

## 与其他模块的关系

- **被** `print_utils.py` 依赖：日志过滤器使用 `match_patterns`
- **被** `image_utils.py` 依赖：使用 `global_once` 控制警告
- **被** `torch_utils.py` 等模块广泛使用
- **独立性强**：该模块本身不依赖项目中的其他工具模块

## 总结

`misc_utils.py` 收集了各种频繁使用但不属于特定领域的工具函数。其中模式匹配系统（`match_patterns`/`filter_patterns`）和事件触发控制（`PeriodicEvent`/`Once`/`global_once`）是训练循环中的高频工具。嵌套属性访问函数在配置系统和模型调试中也有重要应用。这些工具虽然单个功能简单，但组合使用能显著简化训练代码的编写。
