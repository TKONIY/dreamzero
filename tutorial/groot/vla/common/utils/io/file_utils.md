# file_utils.py - 文件系统工具

## 文件概述

`file_utils.py` 提供了一套全面的文件系统操作工具函数，封装了 Python 标准库中 `os`、`shutil`、`glob`、`pickle` 等模块的常用操作，并提供了更简洁、更安全的接口。该模块是整个项目文件 I/O 操作的基础设施层。

模块导出了超过 60 个公开符号（通过 `__all__` 列表），涵盖以下几大类功能：

1. **路径操作**：路径拼接、分割、扩展、判断等
2. **目录操作**：创建、列举、复制、移动、删除
3. **文件读写**：文本、pickle 的加载与保存
4. **归档操作**：tar 包的创建与解压
5. **辅助工具**：MD5 校验、文件锁、时间戳命名等

## 关键代码解析

### 1. 路径拼接核心函数 `f_join`

```python
def f_join(*fpaths):
    fpaths = pack_varargs(fpaths)
    fpath = f_expand(os.path.join(*fpaths))
    if isinstance(fpath, str):
        fpath = fpath.strip()
    return fpath
```

`f_join` 是整个模块中被调用最频繁的函数。它将多个路径片段拼接在一起，同时自动展开 `~`（用户主目录）和环境变量。它还内嵌了 `pack_varargs` 逻辑，既支持 `f_join("a", "b", "c")` 也支持 `f_join(["a", "b", "c"])`。

### 2. 目录列举 `f_listdir`

```python
def f_listdir(*fpaths, filter_ext=None, filter=None, sort=True,
              full_path=False, nonexist_ok=True, recursive=False):
```

一个功能丰富的目录列举函数，支持：
- 按扩展名或自定义函数过滤
- 递归遍历子目录（通过 `os.walk`）
- 目录不存在时返回空列表（而非报错）
- 返回全路径或仅文件名

### 3. 支持 `exist_ok` 的目录复制 `f_copytree`

标准库的 `shutil.copytree` 在目标目录已存在时会报错。该模块重新实现了 `_f_copytree`，添加了 `exist_ok=True` 参数支持目标目录已存在的场景。同时提供了 `_include_patterns` 工厂函数，与 `shutil.ignore_patterns` 互补，用于指定**只复制**匹配的文件。

### 4. 文件名冲突处理

```python
def next_available_file_name(*fpath, suffix_template="_v{i+1}", before_ext=True):
```

当目标文件已存在时，自动在文件名末尾追加版本号后缀（如 `model_v1.pt`、`model_v2.pt`），直到找到不存在的文件名。支持自定义后缀模板和 lambda 回调。

### 5. NFS 安全文件锁

```python
def get_file_lock(*fpath, timeout=15, logging_level="critical"):
```

基于 `flufl.lock` 实现 NFS 安全的文件锁，适用于分布式训练环境中多进程对共享文件系统的并发访问控制。

## 核心类/函数表格

| 函数 | 说明 |
|------|------|
| `f_join(*fpaths)` | 路径拼接，自动展开 `~` 和环境变量 |
| `f_expand(fpath)` | 展开路径中的 `~` 和环境变量 |
| `f_exists(*fpaths)` | 判断路径是否存在 |
| `f_not_empty(*fpaths)` | 判断文件/目录是否非空 |
| `f_mkdir(*fpaths)` | 递归创建目录 |
| `f_mkdir_in_path(*fpaths)` | 为文件路径创建所有父目录 |
| `f_listdir(*fpaths, ...)` | 列举目录内容，支持过滤和递归 |
| `f_copy(fsrc, fdst, ...)` | 复制文件或目录，支持 glob 通配符 |
| `f_copytree(fsrc, fdst, ...)` | 复制目录树，支持 `exist_ok` |
| `f_move(fsrc, fdst)` | 移动文件或目录 |
| `f_remove(*fpath)` | 删除文件或目录，支持 glob |
| `f_glob(*fpath)` | glob 模式匹配文件 |
| `f_split_path(fpath)` | 将路径拆分为组成部分列表 |
| `f_ext` / `f_size` | `os.path.splitext` / `os.path.getsize` 的别名 |
| `f_append_before_ext(fpath, suffix)` | 在扩展名前插入后缀 |
| `f_add_ext(fpath, ext)` | 添加扩展名（如果不存在） |
| `load_pickle` / `dump_pickle` | pickle 文件的读写 |
| `load_text` / `dump_text` | 文本文件的读写 |
| `load_text_lines` / `dump_text_lines` | 按行读写文本 |
| `md5_checksum(*fpath)` | 计算文件 MD5 校验和 |
| `create_tar` / `extract_tar` | tar 归档的创建与解压 |
| `move_with_backup(*fpath)` | 递归备份已存在的文件 |
| `next_available_file_name(...)` | 自动生成不冲突的文件名 |
| `get_file_lock(...)` | 获取 NFS 安全的文件锁 |
| `get_package_root()` | 获取调用者所在 Python 包的根目录 |
| `owner_name(filepath)` | 获取文件所有者名称（Unix） |
| `host_name()` / `host_id()` | 获取主机名/主机 ID |

## 与其他模块的关系

- **依赖** `tree_utils.py`：导入 `is_sequence` 用于参数解包
- **被几乎所有模块依赖**：`f_join` 是项目中最基础的路径工具函数
- **被** `config_utils.py` 使用：`omegaconf_save` 中调用 `f_join`
- **被** `json_utils.py` 使用：所有文件读写函数都通过 `f_join` 处理路径
- **被** `torch_utils.py` 使用：`load_torch` / `save_torch` 中使用 `f_join`

## 总结

`file_utils.py` 是 DreamZero 项目的文件系统操作基础设施，通过统一的接口封装了 Python 标准库的文件操作功能。`f_join` 作为核心路径工具贯穿整个项目。模块特别关注了边界情况的处理（如目录不存在、文件名冲突、NFS 并发访问等），体现了在大规模分布式训练场景下对文件系统操作鲁棒性的需求。模块还提供了丰富的别名（如 `pickle_load` = `load_pickle`），以适应不同的命名习惯。
