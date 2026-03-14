# json_utils.py - JSON/YAML 数据工具

## 文件概述

`json_utils.py` 提供了 JSON、YAML 以及 JSONL（JSON Lines）格式数据的读写工具。该模块的设计原则是：

1. 统一接口风格（`json_load`/`json_dump`、`yaml_load`/`yaml_dump` 等）
2. 支持路径自动拼接（通过 `f_join`）
3. 支持自动类型转换（将 NumPy/PyTorch 对象转为 JSON 可序列化的原生类型）
4. 提供根据文件扩展名自动选择格式的便利函数
5. 提供 `Jsonl` 类实现 JSONL 文件的流式追加写入

## 关键代码解析

### 1. 基础读写函数

```python
def json_load(*file_path, **kwargs):
    file_path = f_join(file_path)
    with open(file_path, "r") as fp:
        return json.load(fp, **kwargs)
```

所有读写函数都遵循统一模式：路径参数支持可变参数形式（如 `json_load("dir", "file.json")`），内部通过 `f_join` 拼接。`**kwargs` 透传给底层的标准库函数。

### 2. 自动类型转换 `any_to_primitive`

```python
@make_recursive_func
def any_to_primitive(x):
    if isinstance(x, (np.ndarray, np.number, torch.Tensor)):
        return x.tolist()
    else:
        return x
```

使用 `@make_recursive_func` 装饰器，自动将嵌套数据结构中的 NumPy 数组和 PyTorch 张量转换为 Python 原生类型（list/float/int）。这是 `convert_to_primitive=True` 选项的底层实现。

### 3. YAML 写入的默认参数

```python
def yaml_dump(data, *file_path, dumper=yaml.safe_dump, convert_to_primitive=False, **kwargs):
    indent = kwargs.pop("indent", 2)
    default_flow_style = kwargs.pop("default_flow_style", False)
    sort_keys = kwargs.pop("sort_keys", False)  # 保留原始字典顺序
```

YAML 写入默认使用 2 空格缩进、块状样式（非 flow 样式）、保留原始键序。这些默认值产生人类可读的配置文件。

### 4. 自动格式识别

```python
def json_or_yaml_load(*file_path, **loader_kwargs):
```

根据文件扩展名（`.json`、`.yml`、`.yaml`）自动选择加载器，简化调用者的代码。

### 5. Jsonl 类

```python
class Jsonl:
    def __init__(self, *file_path, mode="a"):
```

`Jsonl` 提供了 JSONL 格式的流式读写能力：

- 支持三种模式：`r`（只读）、`w`（覆盖写）、`a`（追加写）
- `append()` 方法立即 flush 到磁盘，适合训练日志等需要实时持久化的场景
- 实现了上下文管理器协议（`with` 语句）和容器协议（`len`、`iter`、`getitem`）
- 在 `a` 模式下，会先加载已有数据到内存，实现"内存+磁盘"双缓冲

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `json_load(*file_path)` | 从文件加载 JSON |
| `json_loads(string)` | 从字符串解析 JSON |
| `json_dump(data, *file_path)` | 将数据写入 JSON 文件 |
| `json_dumps(data)` | 将数据序列化为 JSON 字符串 |
| `jsonl_load(*file_path)` | 加载 JSONL 文件为列表 |
| `jsonl_dump(data, *file_path)` | 将列表写入 JSONL 文件 |
| `yaml_load(*file_path)` | 从文件加载 YAML |
| `yaml_loads(string)` | 从字符串解析 YAML |
| `yaml_dump(data, *file_path)` | 将数据写入 YAML 文件 |
| `yaml_dumps(data)` | 将数据序列化为 YAML 字符串 |
| `json_or_yaml_load(*file_path)` | 根据扩展名自动加载 JSON 或 YAML |
| `json_or_yaml_dump(data, *file_path)` | 根据扩展名自动保存为 JSON 或 YAML |
| `any_to_primitive(x)` | 递归将 NumPy/Tensor 转换为原生 Python 类型 |
| `Jsonl` 类 | JSONL 文件的流式读写器 |

所有函数都有以 `load_`/`dump_` 开头的别名（如 `load_json` = `json_load`）。

## 与其他模块的关系

- **依赖** `file_utils.py`：所有路径操作通过 `f_join` 处理
- **依赖** `functional_utils.py`：使用 `make_recursive_func` 装饰器实现递归类型转换
- **被配置系统使用**：项目中的 YAML 配置文件通过这些函数加载和保存
- **被数据处理流程使用**：JSON/JSONL 格式常用于数据集元信息和训练日志

## 总结

`json_utils.py` 统一了 JSON 和 YAML 两种常用数据格式的读写接口，通过支持路径可变参数和自动类型转换，降低了使用门槛。`Jsonl` 类特别适合训练过程中的日志记录场景——每条记录立即持久化，同时保持内存中可查询的完整数据。自动格式识别功能让调用者无需关心具体的文件格式。
