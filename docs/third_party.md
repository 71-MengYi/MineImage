# 第三方节点编辑器方案对比

MineImage 的节点画布是自己用 `QGraphicsView / QGraphicsItem` 实现的。本文记录
选型时比较过的开源方案、各自取舍，以及为什么最终没有直接依赖现成节点编辑器。

> **数据说明**：下表中的许可证、星标数、最近推送时间来自 **2026-10-21** 通过
> GitHub REST API 查询的快照，仅用于判断活跃度，数字会随时间变化；没有查到可靠
> 数据的字段写“未核实”，不臆造版本号或星标数。许可证以各仓库的 `LICENSE` 为准。

---

## 1. 候选方案速览

| 项目 | 形态 / 技术栈 | 许可证 | 活跃度（2026-10-21 查询） | 嵌入 PySide6 应用的成本 |
| --- | --- | --- | --- | --- |
| [NodeGraphQt](https://github.com/jchanvfx/NodeGraphQt) | Python + Qt 节点图 UI 框架；原始 README 面向 PySide2，社区有 PySide6 分支 | MIT | 1835★；最近推送 2026-03-11；未归档 | 中：自带节点/端口/连线/属性面板，但节点模型、序列化、撤销需要适配 |
| [Ryven](https://github.com/leon-thomm/Ryven) | Python + Qt 的流式可视化编程环境/框架 | MIT | 4092★；最近推送 2026-08-02；**仓库已归档（archived=true）** | 高：更偏“完整应用/脚本环境”，长期维护有风险 |
| [qtpynodeeditor](https://github.com/klauer/qtpynodeeditor) | Python + `qtpy`（PyQt5/PySide）节点编辑器；Qt5 时代代码 | BSD-3-Clause | 252★；最近推送 2024-12-15 | 中高：连线/端口控件完整，但无数据流引擎、无序列化、PySide6 未在 README 声明 |
| [QtNodes / nodeeditor](https://github.com/paceholder/nodeeditor) | C++ + Qt Widgets 数据流图库（QtNodes 3.0） | BSD-3-Clause | 3714★；最近推送 2026-07-31 | 高：功能最完整，但官方是 C++ 库，Python 侧需自行绑定/封装 |
| [QNodeEditor](https://github.com/JasperJeuken/QNodeEditor) | Python + PyQt5 节点编辑控件（[PyPI](https://pypi.org/project/QNodeEditor/) 1.0.7，2026-05-11） | BSD-3-Clause（PyPI classifier 误写为 MIT） | 41★；PyPI 仍在更新 | 中高：依赖 PyQt5，不是 PySide6；需要跨绑定移植 |
| [Nodezator](https://github.com/IndieSmiths/nodezator) | Python + pygame-ce 的通用节点式应用 | Unlicense | 2792★；最近推送 2026-10-20 | 很高：是独立应用而不是 Qt 框架，技术栈不兼容 |
| [PyFlow](https://github.com/pedroCabrera/PyFlow) | Python + PySide2 的可视化脚本框架（节点/子图） | Apache-2.0 | 3202★；最近推送 2025-09-02 | 高：PySide2 时代，框架体积大，和现有 PySide6 应用整合成本高 |

其他相关项目（例如 `pyqt-node-editor`、`NodeEditor` 的各类 fork）多为上述项目
的前身或衍生版本，能力和许可证接近，未单列。

---

## 2. 逐个说明

### 2.1 NodeGraphQt

* 仓库：<https://github.com/jchanvfx/NodeGraphQt>，PyPI：<https://pypi.org/project/NodeGraphQt/>
* 许可证：MIT（GitHub API 返回 `spdx_id: MIT`）。
* 优点：节点、端口、管道、属性面板、节点面板、框选、缩放等 UI 齐全；文档和
  示例较完整；MIT 对商业/闭源分发友好。
* 局限：它是“节点图 UI 框架”，不是图像处理引擎；节点参数控件、DAG 执行、
  中间结果缓存、撤销语义仍要自己做。原始版本以 PySide2 为主，PySide6 主要靠
  社区 fork（例如 `NodeGraphQt-PySide6`），需要评估分支维护情况。
* 对 MineImage：如果只是想要“更成熟的连线绘制”，它是首选候选；但会引入第二套
  节点模型，和现有 `GraphModel` / `QUndoStack` / 引擎 revision 语义重复。

### 2.2 Ryven

* 仓库：<https://github.com/leon-thomm/Ryven>，文档：<https://leon-thomm.github.io/Ryven/>
* 许可证：MIT。
* 优点：流式可视化编程，节点/执行/保存等概念完整；社区关注度高。
* 局限：更接近一个独立的 Python 可视化编程环境；GitHub API 当前标记
  `archived: true`，意味着上游不再接受常规维护。把归档项目作为长期依赖风险较高。
* 对 MineImage：适合借鉴交互与执行模型，不适合作为基础库。

### 2.3 qtpynodeeditor

* 仓库：<https://github.com/klauer/qtpynodeeditor>，文档：<https://klauer.github.io/qtpynodeeditor/>
* 许可证：BSD-3-Clause（LICENSE 中明确为 NodeEditor 的衍生作品）。
* 优点：纯 Python，基于 `qtpy` 可切换 PyQt5/PySide；节点、端口、连接、场景
  已经封装好；代码量相对小，便于阅读和改。
* 局限：面向 Qt5 时代；README 只声明 `qtpy, PyQt5, PySide`，没有声明 PySide6；
  不提供图模型、序列化、数据流引擎、参数面板；最后一次推送是 2024-12。
* 对 MineImage：可以作为“快速搭一个节点画布”的选项，但在 PySide6 + Python 3.13
  下需要自己解决兼容性，且仍要补全模型/引擎/参数 UI。

### 2.4 QtNodes（paceholder/nodeeditor）

* 仓库：<https://github.com/paceholder/nodeeditor>，文档：<https://qtnodes.readthedocs.io/>
* 许可证：BSD-3-Clause。
* 优点：C++/Qt 生态里最成熟的节点图库之一；QtNodes 3.0 提供 Model-View 架构、
  DataFlowGraphModel、序列化、Undo/Redo、样式定制，功能覆盖最全；文档完善。
* 局限：官方仓库是 **C++** 库；要在 Python/PySide6 中使用，需要 pybind11 /
  Shiboken 之类的绑定层或独立进程/插件方案，构建、分发、调试成本高。官方文档
  和示例以 C++ 为主，未提供官方 Python 包（第三方绑定质量需自行验证）。
* 对 MineImage：如果未来需要 C++ 性能核心、复杂子图/状态机，QtNodes 是强候选；
  但当前项目是纯 Python + small team，引入 C++ 构建链不划算。

### 2.5 QNodeEditor

* 仓库：<https://github.com/JasperJeuken/QNodeEditor>，PyPI：<https://pypi.org/project/QNodeEditor/>
* 许可证：**BSD-3-Clause**。GitHub API 返回 `spdx_id: BSD-3-Clause`，仓库
  `LICENSE` 也是 BSD-3-Clause 条款；PyPI 的 classifier 仍写着 “MIT License”，
  属于元数据错误，使用时应以仓库 LICENSE 为准。
* 优点：API 简洁，节点内可放任意 Qt 控件，场景异步求值，适合做小型节点工具；
  文档和示例清晰；PyPI 1.0.7 于 2026-05-11 发布，仍在维护。
* 局限：依赖 **PyQt5**（`PyQt5~=5.15.10`）和 `networkx`，不是 PySide6；在
  PySide6 应用中直接使用会遇到绑定不兼容，需要移植或双绑定共存。
* 对 MineImage：如果项目本来就用 PyQt5，会是不错的选择；但 MineImage 已统一
  PySide6，迁移代价大于自绘成本。

### 2.6 Nodezator

* 仓库：<https://github.com/IndieSmiths/nodezator>，官网：<https://nodezator.com/>
* 许可证：Unlicense（公共领域）。
* 优点：把任意 Python callable 变成节点非常方便，可导出 Python 代码；社区活跃。
* 局限：基于 **pygame-ce** 的独立桌面应用，不是 Qt 控件/框架；无法嵌入
  PySide6 主窗口，也没有 Qt 的样式、Dock、高 DPI 体系。
* 对 MineImage：不适用；仅可借鉴“函数即节点”的产品思路。

### 2.7 PyFlow

* 仓库：<https://github.com/pedroCabrera/PyFlow>
* 许可证：Apache-2.0。
* 优点：节点/子图/可视化脚本框架较完整，Apache-2.0 友好。
* 局限：基于 PySide2；体量大；将现有 PySide6 应用与它整合需要处理两套 Qt 绑定
  和事件循环；项目更新节奏不稳定。
* 对 MineImage：不采用；可参考其子图和节点注册设计。

---

## 3. 为什么 MineImage 选择自绘 QGraphicsView

### 3.1 模型必须是唯一事实来源

MineImage 的核心是 `GraphModel`（节点、端口、连线、revision）和异步执行引擎。
引擎用 **generation + revision** 丢弃过期结果；撤销用 `QUndoStack` 命令；工程
文件直接序列化模型。现成节点编辑器通常自带一套节点/连线/撤销/序列化模型，接入
后会出现“两套真相”，每次同步都可能引入竞态或撤销不一致。

自绘画布把 `GraphView` 做成模型的投影：模型变更 → 重建/更新图元；用户操作 →
`QUndoCommand` 修改模型。这条链路短、可测试，也方便脚本/测试直接驱动。

### 3.2 图像工作流有大量专用 UI

图像处理场景需要一些通用节点编辑器不会提供的能力：

* 节点上显示**结果缩略图**、错误标记、计算中状态、耗时；
* 双击节点切换到**参数面板**，而不是弹属性窗口；
* 右侧预览的**单视图/左右对比/滑动对比**、像素采样、点/矩形/线拾取；
* 拖到连线上**自动插入节点**并重新布局；
* 纵向自动布局（父节点对齐子节点中心）；
* 三栏 1:1:4 布局、中文界面、统一 QSS 主题。

这些逻辑大部分在节点图库之外，用自绘画布反而更容易保持一致。

### 3.3 依赖与打包

* 只需要 PySide6 + NumPy + OpenCV + Pillow；PyInstaller onefile 配置简单。
* 不引入额外的节点图框架、Qt5/PySide2 兼容层或 C++ 构建链。
* 许可证面更窄：PySide6 自身按 LGPLv3/GPLv3/商业许可发布（见
  <https://doc.qt.io/qtforpython-6/licenses.html>），再叠加第三方库时需要逐一
  评估分发义务；自绘没有新增第三方许可证。

### 3.4 可测试性与可维护性

`tests/test_graph.py`、`tests/test_integration.py` 可以在 offscreen 模式下直接
构造 `GraphView`、模拟端口拖拽、检查撤销栈和自动布局。自绘代码虽多，但边界清晰、
测试覆盖到位；引入外部框架则需要额外 adapter 和 mock，测试成本反而更高。

### 3.5 成本与风险

自绘的代价是我们自己维护：

* 连线路径/命中测试/端口拖拽；
* 节点绘制、缩略图、选择与移动；
* 序列化与撤销语义；
* 自动布局。

这些代码量可控（`app/graph/` 约一千多行），并且已经由测试覆盖。相比之下，
适配一个外部节点编辑器需要理解其模型、绕开其假设、维护 fork/绑定，长期成本更高。

### 3.6 什么时候应该重新评估

如果出现以下需求，建议重新评估 QtNodes / NodeGraphQt：

* 子图（subgraph）、节点组折叠、跨图复制粘贴；
* 多人协作/实时同步编辑；
* 需要 C++ 级渲染性能或十万级节点；
* 需要成熟的属性编辑器、节点面板、主题市场。

在当前“单机、图像处理实验、算子可扩展”的定位下，自绘 QGraphicsView 是更合适
的工程取舍。

---

## 4. 参考链接

* NodeGraphQt：<https://github.com/jchanvfx/NodeGraphQt> ·
  <https://pypi.org/project/NodeGraphQt/>
* Ryven：<https://github.com/leon-thomm/Ryven> ·
  <https://leon-thomm.github.io/Ryven/>
* qtpynodeeditor：<https://github.com/klauer/qtpynodeeditor> ·
  <https://klauer.github.io/qtpynodeeditor/>
* QtNodes：<https://github.com/paceholder/nodeeditor> ·
  <https://qtnodes.readthedocs.io/>
* QNodeEditor：<https://github.com/JasperJeuken/QNodeEditor> ·
  <https://pypi.org/project/QNodeEditor/>
* Nodezator：<https://github.com/IndieSmiths/nodezator> ·
  <https://nodezator.com/>
* PyFlow：<https://github.com/pedroCabrera/PyFlow>
* Qt for Python 许可：<https://doc.qt.io/qtforpython-6/licenses.html>
