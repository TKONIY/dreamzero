# termcolor.py - 终端彩色输出

## 文件概述

`termcolor.py` 提供了终端 ANSI 彩色文本输出功能。该文件基于 Konstantin Lepa 的开源实现（MIT 许可证），经过适配后集成到 DreamZero 项目中。它提供了两个核心函数：`color_text`（生成带颜色的字符串）和 `cprint`（直接打印彩色文本）。

## 关键代码解析

### 1. 颜色和样式定义

模块通过三个字典定义了所有可用的终端样式：

- **`COLORS`**：前景色，ANSI 码 30-37，包括 grey、red、green、yellow、blue、magenta、cyan、white
- **`HIGHLIGHTS`**：背景色，ANSI 码 40-47，颜色名称与 `COLORS` 相同
- **`STYLES`**：文字样式，ANSI 码 1-8，包括 bold、dark、underline、blink、reverse、concealed

### 2. 颜色文本生成

```python
def color_text(text, color=None, bg_color=None, styles=None):
```

通过 ANSI 转义序列 `\033[<code>m` 包裹文本实现着色。多种属性可以叠加（前景色 + 背景色 + 多个样式）。最后追加 `\033[0m`（`RESET`）重置终端状态。

如果环境变量 `ANSI_COLORS_DISABLED` 存在，函数直接返回原始文本，这为非终端环境（如日志文件）提供了兼容性。

### 3. 彩色打印

```python
def cprint(*args, color=None, bg_color=None, styles=None, **kwargs):
```

`cprint` 先将所有参数通过 `print()` 写入 `StringIO` 缓冲区得到完整文本，然后对整个文本应用颜色，最后输出。这样可以完整支持 `print()` 的参数格式。

## 核心类/函数表格

| 名称 | 说明 |
|------|------|
| `COLORS` | 前景色字典：`{"grey": 30, "red": 31, ...}` |
| `HIGHLIGHTS` | 背景色字典：`{"grey": 40, "red": 41, ...}` |
| `STYLES` | 文字样式字典：`{"bold": 1, "underline": 4, ...}` |
| `RESET` | ANSI 重置码 `"\033[0m"` |
| `color_text(text, color, bg_color, styles)` | 返回添加了 ANSI 颜色码的字符串 |
| `cprint(*args, color, bg_color, styles)` | 直接打印彩色文本 |

## 与其他模块的关系

- **独立模块**：不依赖项目中的其他模块
- **通过** `io/__init__.py` 导出：可以从 `groot.vla.common.utils` 直接导入 `cprint` 和 `color_text`
- **用途**：在训练脚本、调试输出中提供视觉区分，如用红色高亮错误、绿色表示成功等

## 总结

`termcolor.py` 是一个轻量级的终端着色工具，提供了 8 种文字颜色、8 种背景色和 6 种文字样式的组合能力。通过环境变量 `ANSI_COLORS_DISABLED` 可以全局禁用颜色输出，保证在非终端环境下的兼容性。模块接口简洁——`color_text` 返回着色字符串，`cprint` 直接打印，覆盖了绝大多数使用场景。
