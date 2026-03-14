# functional_utils.py - 函数式编程与元编程工具

## 文件概述

`functional_utils.py` 是整个 `common/utils` 模块中最基础的工具文件之一，提供了大量函数式编程和元编程工具。该模块的核心功能包括：

1. **元装饰器**：`meta_decorator`，让装饰器可以省略括号
2. **递归函数生成器**：`make_recursive_func`，将单元素函数升级为支持嵌套结构的函数
3. **函数生命周期控制**：`deprecated`（弃用标记）、`call_once`（仅调用一次）
4. **空操作占位**：`NoopObject`、`NoopContext`
5. **类注册系统**：`make_registry_metaclass`、`ClassRegistry`
6. **参数检查与打包**：`pack_varargs`、`pack_kwargs`、`has_keys` 等
7. **函数签名工具**：参数检查、签名兼容性验证

## 关键代码解析

### 1. `meta_decorator` -- 元装饰器

```python
def meta_decorator(decor):
```

这是一个"装饰器的装饰器"，解决了 Python 装饰器的一个常见痛点：当装饰器有可选参数时，使用者需要区分是否加括号。`meta_decorator` 让被装饰的装饰器同时支持两种用法：

```python
@my_decorator          # 无参数
def func(): ...

@my_decorator(arg=1)   # 有参数
def func(): ...
```

实现原理：检查传入参数是否是"单个可调用对象"。如果是，说明装饰器被无括号调用；否则，返回一个接受被装饰函数的 lambda。

### 2. `make_recursive_func` -- 递归函数生成器

```python
@meta_decorator
def make_recursive_func(fn, *, with_path=False):
```

这是整个工具库中最重要的装饰器之一。它将一个只处理单个叶节点的函数，升级为能自动遍历任意嵌套数据结构的函数：

```python
@make_recursive_func
def to_cuda(x):
    return x.cuda()

# 现在可以处理嵌套字典：
data = {"obs": {"image": tensor1, "state": tensor2}, "action": tensor3}
data_gpu = to_cuda(data)  # 所有张量都移到 GPU
```

内部使用 `tree.map_structure`（来自 DeepMind 的 `dm-tree` 库）实现结构遍历。`with_path=True` 时还会传递当前节点在树中的路径。

### 3. `call_once` -- 单次调用控制

```python
@meta_decorator
def call_once(func, on_second_call="noop"):
```

确保函数只被执行一次。第二次及以后的调用会根据 `on_second_call` 参数：
- `"noop"`：静默忽略
- `"raise"`：抛出异常
- `"warn"`：发出警告

典型用法是注册操作（如 `register_omegaconf_resolvers`），避免重复注册。

### 4. 类注册系统

模块提供了两种类注册机制：

**基于元类的注册**：
```python
TrainerRegistry = make_registry_metaclass('TrainerRegistry')
class BaseTrainer(metaclass=TrainerRegistry):
    pass
```
所有继承 `BaseTrainer` 的子类会自动注册到 `TrainerRegistry.registry` 字典中。

**基于实例的注册**：
```python
class BaseClass:
    registry = ClassRegistry()
    def __init_subclass__(cls, **kwargs):
        cls.registry.add(cls)
```
`ClassRegistry` 使用 `__init_subclass__` 钩子，兼容 pickle 和 Ray 等序列化工具。

### 5. 参数打包工具

```python
def pack_varargs(args):   # 将 *args 或单个列表统一为列表
def pack_kwargs(args, kwargs):  # 将 **kwargs 或单个字典统一为字典
```

这些函数让 API 同时支持 `f(a, b, c)` 和 `f([a, b, c])` 两种调用形式。

### 6. `state_dict_class` -- 状态序列化装饰器

```python
def state_dict_class(keys: list[str]):
```

类装饰器，为任意 Python 类添加 PyTorch 风格的 `state_dict()` 和 `load_state_dict()` 方法。指定需要序列化的属性名列表即可。

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `meta_decorator(decor)` | 让装饰器支持有/无括号两种用法 |
| `make_recursive_func(fn)` | 将单元素函数升级为支持嵌套结构 |
| `deprecated(func, msg, action)` | 标记函数为已弃用 |
| `call_once(func, on_second_call)` | 确保函数只执行一次 |
| `state_dict_class(keys)` | 为类添加 state_dict 方法 |
| `implements_method(obj, method)` | 检查对象是否实现指定方法 |
| `NoopObject` | 调用任何方法都无操作的对象 |
| `NoopContext` | 什么都不做的上下文管理器 |
| `make_registry_metaclass(name)` | 创建自动注册子类的元类 |
| `ClassRegistry` | 基于实例的类注册表 |
| `pack_varargs(args)` | 统一可变参数为列表 |
| `pack_kwargs(args, kwargs)` | 统一关键字参数为字典 |
| `enable_list_arg(func)` | 让 *args 函数也接受列表参数 |
| `enable_varargs(func)` | 让列表参数函数也接受 *args |
| `merge_kwargs(args, kwargs)` | 合并所有字典参数和关键字参数 |
| `func_parameters(func)` | 获取函数参数签名 |
| `func_has_arg(func, arg_name)` | 检查函数是否有指定参数 |
| `is_signature_compatible(func, *args)` | 检查调用是否兼容函数签名 |
| `make_list(x)` / `make_tuple(elem, repeats)` | 标量到序列的转换 |
| `accumulate(iterable, fn)` | 累积计算（类似 itertools.accumulate） |
| `DecoratorContextManager` | 让上下文管理器同时可用作装饰器 |
| `method_decorator(decorator)` | 将普通函数装饰器转为方法装饰器 |
| `has_keys(D, keys)` / `assert_has_keys(D, keys)` | 字典键检查 |

## 与其他模块的关系

- **依赖** `tree_utils.py`：使用 `is_mapping`、`is_sequence`
- **被几乎所有其他工具模块依赖**：
  - `config_utils.py` 使用 `call_once`、`meta_decorator`、`is_mapping`、`is_sequence`
  - `json_utils.py` 使用 `make_recursive_func`
  - `print_utils.py` 使用 `meta_decorator`
  - `array_tensor_utils.py` 使用 `make_recursive_func`
  - `torch_utils.py` 使用 `implements_method`、`accumulate`

## 总结

`functional_utils.py` 是整个工具库的元编程基础。`meta_decorator` 和 `make_recursive_func` 这两个核心工具被大量其他模块使用——前者统一了装饰器的调用语法，后者解决了"对嵌套数据结构批量操作"这一深度学习和机器人学习中的核心需求。类注册系统为配置驱动的对象构造提供了底层支持。整体而言，该模块通过高阶函数和元编程技术，显著减少了项目中的样板代码。
