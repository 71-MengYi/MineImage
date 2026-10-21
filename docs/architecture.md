# MineImage 架构与模块边界

MineImage 是一个基于 PySide6 的节点式图像处理实验工作台。本文件描述**模块职责与
跨团队接口**，实现细节见各模块 docstring。

## 总体数据流

```
文件/菜单 ──> MainWindow ──> WorkspaceWidget（每个标签页一个）
                                  ├── OperatorLibrary  左侧算子库（搜索/整行折叠一级分类/多级折叠树/拖拽/说明浮层）
                                  ├── GraphView        中间节点画布  ←→ GraphModel
                                  ├── ParamPanel       中间叠层（双击节点进入）
                                  ├── PreviewWidget    右侧预览（缩放/平移/对比/交互拾取）
                                  └── ExecutionEngine  后台计算 + 节点快照
                                           ↑
                                  OperatorRegistry（自动扫描 app/operators/builtin + 用户目录）
```

* **模型**：`app/core/model.py`。`WorkspaceModel` 持有 `GraphModel` 与源图像
  `images: dict[str, ImageData]`。节点参数变化通过 `NodeModel.touch()` 增加
  `revision`。引擎用 revision 丢弃过期结果。
* **端口数据类型**：`app/core/data_types.py` 是端口类型的唯一来源。`DataType`
  定义 `image / number / edges / contours / components` 五种类型；
  `NumberValue / EdgesData / ContoursData / ComponentsData` 是结构化值的运行时
  容器；`coerce_data_type()` 归一化中英文别名，`render_preview()` /
  `summarize_value()` 供算子显式可视化与摘要使用。`PortSpec.data_type` 声明端口类型
  （默认图像），`port_types_compatible() / connection_error()` 决定连线是否允许，
  `validate_inputs()` 在执行前校验输入，`output_to_dict() / coerce_output_value()`
  按端口类型规范化算子返回值。
* **类型化数据流**：数据流从“图像容器”升级为“类型化端口值”。连线传递的不再只
  是 `ImageData`，也可能是 `NumberValue`、`EdgesData`、`ContoursData` 或
  `ComponentsData`；`NodeResult.outputs` 是泛化的 `dict[str, Any]`，
  `preview_image` 优先取图像输出、否则渲染结构化数据，`summary_text()` 汇总
  各端口摘要。
* **结构化数据不再自动预览**：`NodeResult.preview_image` 只返回图像输出；数值、
  边缘、轮廓、连通域不再由 `render_preview()` 转成主预览或节点缩略图。节点卡片对
  单个数值直接显示、对图像显示缩略图、对数组提示「数组数据 / 双击查看」；参数面板的
  「输出数据」区在结果包含非图像输出时自动出现，以文本详情展示数组的数量、画布尺寸、
  形状、前几个点等，不再生成预览图。
* **万能端口与输出节点**：`PortSpec(any_type=True)` 表示万能端口，输入可接收五类
  数据中的任意一种，输出按运行时类型原样透传；连线校验只要任意一端为万能端口即兼容。
  `其他 → 可视化 → 输出预览`（`output.display`）就用一个 `data` 输入和一个
  `data` 输出实现任意链路的终点。
* **算子**：`app/operators/spec.py` 定义 `OperatorSpec / ParamSpec / PortSpec /
  Operator`；`loader.py` 扫描目录。每个算子文件定义模块级 `OPERATOR`。
  参数联动 `visible_when/enabled_when` 使用受限 `eval`：支持裸参数名与
  `params['key']`，**不支持** `params.key` 属性访问；表达式异常时按 True 处理。
* **引擎**：`app/engine/engine.py`。每次请求在主线程抓取一份不可变的图快照，再在
  后台线程池中按 DAG 拓扑序提交节点任务：兄弟分支并行执行，有依赖的节点等待上游
  future 完成；`update_nodes` / `preview_node` / `commit_node` 递归重算目标节点与
  全部下游，`refresh_node` 只重算单个节点。结果按 `generation + revision` 双重
  校验丢弃过期写回，节点错误按分支隔离。算子 `process()` 会被多个线程并发调用，
  必须线程安全（无实例可变状态、不原地修改输入）。默认并行度为
  `min(4, CPU 核数)`（1–16，`execution/workers` 可调）。发出
  `node_started/node_finished/node_failed/chain_finished/busy_changed`。
  `NodeResult.outputs` 可保存任意类型的端口值；`preview_image` 与
  `summary_text()` 负责预览与摘要，不再假定输出一定是图像。
* **画布**：`app/graph/*`，QGraphicsView 自绘。所有图修改走 `QUndoStack`。
  端口改为上输入 / 下输出，连线使用纵向三次贝塞尔曲线（`app/graph/curve.py`）；
  拖拽落点靠近端口时自动连线、落在连线上时自动插入节点，移动与连线命令合并为
  同一个撤销宏。`PortItem` 按 `DataType` 着色并过滤可连接端口：只有同类型端口
  才能建立连线，图像端口还要求颜色空间有交集。
* **界面**：`app/ui/*`，主窗口与工作区负责组装，其余控件保持可独立测试。
* **ONNX 模型下载**：`app/core/models.py` 维护 OpenCV Zoo 模型清单（YuNet / SFace /
  MobileNetV2）、本地缓存目录与下载逻辑（进度 / 取消 / 原子落盘 / HTML 与 SHA-256
  校验）；
  `app/ui/model_download.py` 用 Qt 线程池把下载放到后台，参数面板与设置对话框
  共用。默认自动下载可用 `MINEIMAGE_AUTO_DOWNLOAD_MODELS=0` 关闭，缓存目录可用
  `MINEIMAGE_MODEL_DIR` 覆盖。
* **参数界面扩展**：`app/ui/param_widgets.py` 提供 `register_param_widget` /
  `register_param_layout` 注册表，`ParamSpec.widget` / `ParamSpec.layout` /
  `ParamSpec.widget_options` 只替换编辑器与行布局，不改变参数的校验与运行时值。
  内置 `model_file` 控件把模型下载状态、官方地址与进度直接放进参数面板。

## 端口数据类型系统

`app/core/data_types.py` 定义节点之间传递的数据种类；`app/operators/spec.py`
把它们落到端口声明、连线校验与返回值规范化上。

| `DataType` | 值 | 运行时容器 | 说明 |
| --- | --- | --- | --- |
| `IMAGE` | `image` | `ImageData` | 默认端口类型；`ndarray` 自动包装并按形状推断颜色空间 |
| `NUMBER` | `number` | `NumberValue` | 携带 `label` / `unit`；也可直接返回 `int` / `float` |
| `EDGES` | `edges` | `EdgesData` | `segments` 为 `(N,2)` 点数组或点数组元组，可带 `image_size` |
| `CONTOURS` | `contours` | `ContoursData` | 与 `EdgesData` 同构，另可携带 `hierarchy` |
| `COMPONENTS` | `components` | `ComponentsData` | 二维 int32 标签图（0 为背景）+ 面积 / 质心；也可直接返回标签图 |

关键约束：

* 端口类型由 `PortSpec.data_type` 声明，可写枚举或中英文别名（`coerce_data_type`
  归一化）；默认 `DataType.IMAGE`。
* `accepts` 只对图像端口有效；非图像端口构造时会被清空，不参与颜色空间校验。
* 连线类型由 `port_types_compatible(output_port, input_port)` 判定：两端
  `data_type` 必须一致；图像端口在双方都声明 `accepts` 时要求颜色空间有交集。
  失败时用 `connection_error()` 生成中文提示。
* 执行前 `validate_inputs()` 再次检查输入类型与颜色空间；执行后
  `output_to_dict()` 按输出端口类型调用 `coerce_output_value()` 规范化返回值，
  不合法时报 `输出“xxx”的数据无效：...`。
* 旧 `.mi` 工程仍可打开，历史连线不做回溯校验；新建 / 重新连线时才校验类型。
* 结构化数据不再自动渲染成主预览 / 缩略图；`render_preview()` 仅供算子显式
  生成可视化输出。`summarize_value()` / `summarize_outputs()` 生成状态栏与节点摘要，
  数组详情在参数面板的「输出数据」区以文本展示。

## 流程框节点与自定义算子

流程框（Frame Node）是画布上的容器节点：自身带动态数量的输入 / 输出端口，内部可以放普通
算子节点，也可以继续嵌套流程框；整个框可保存成「用户自定义算子」，再从算子库拖回画布复用。
实现说明见 [frame_node.md](frame_node.md)，用户操作见 [user_guide.md](user_guide.md) 第 8 节，
算子文件写法见 [operator_guide.md](operator_guide.md) 第 12 节。

### 数据模型

* **扁平图 + parent_frame 链**：框内节点仍在同一个 `GraphModel.nodes` 里，通过
  `NodeModel.parent_frame` 指向所属框；嵌套框也只是多一级链。`GraphModel` 提供
  `direct_children / frame_subtree / is_inside / node_container` 查询，不引入第二套图结构。
* **NodeModel.frame**：只有流程框节点（`op_id == FRAME_OP_ID`，即 `"__frame__"`）使用，
  `is_frame` 为 `op_id == FRAME_OP_ID and frame is not None`。
* **FrameModel / FramePort**（`app/core/frame.py`）：框的输入、输出端口列表与来源
  `operator_id`。每侧始终保留一个待定端口（`data_type=None`，对应
  `PortSpec(any_type=True)`）；连线确定类型后 `FrameModel.bind()` 把待定端口定类型并
  `ensure_pending()` 追加新的待定端口。
* **工程序列化**：`NodeModel.to_dict()` 写出 `parent_frame`（非空时）与 `frame`；
  `GraphModel.from_dict()` 加载后修复悬空父框（父节点不存在 / 不是流程框时回退顶层）。

### 端口层级与内外侧

流程框的每个端口在画布上同时有外侧与内侧两个圆形连接点（内侧向内偏移 14px，图形相同）。
模型用 `GraphModel.port_level(node_id, port_key, role)` 定义层级，只有层级相同才允许连线：

| 端口 | 作为源（`src`） | 作为目标（`dst`） |
| --- | --- | --- |
| 普通节点端口 | 节点的 `parent_frame` | 节点的 `parent_frame` |
| 流程框输入端口 | 框 id（内侧，框内数据源） | 框的 `parent_frame`（外侧，接收框外数据） |
| 流程框输出端口 | 框的 `parent_frame`（外侧，向框外供数） | 框 id（内侧，接收框内数据） |

* `connection_role()` 把连线归类为 `normal / frame_input_inner / frame_output_inner /
  frame_input_outer / frame_output_outer / invalid`；
* `connection_allowed()` 在节点、端口存在性之外做层级校验，失败时返回中文原因；
* 回环检测 `effective_successors()` 忽略 `frame_input_inner` 边界边，避免把
  「框输入 → 框内节点 → 框输出」误判为环；
* `resolve_source(src_node, src_port)` 递归跨过框边界，解析出真正的普通节点端口：
  框输入端口沿外侧入边解析，框输出端口沿内侧入边解析，断开时返回 `None`；
  `frame_output_sources(frame_id)` 返回每个输出端口的真实来源，供引擎使用。

### 画布与交互

* `app/graph/frame_item.py` 的 `FrameItem` 绘制圆角容器、标题栏与「待定 / 依赖缺失」徽标；
  外框输入端口在顶边、输出端口在底边，内侧端口在框内 14px 处。
* 框几何不单独保存：`GraphView._sync_frame_geometries()` 按**直接子节点**包围盒 + padding
  反推框的位置与尺寸（无子节点默认 420×220）；嵌套框按深度由内向外同步。
* 空白处右键「添加流程框」；从算子库拖入框区域时取最深命中的嵌套框作为 `parent_frame`；
  拖动普通节点按中心命中的框更新所属关系；拖动框体整体平移全部后代。
* 删除框会连同全部后代与相关连线一步撤销；复制 / 粘贴框会带上后代与内部连线并重映射
  `parent_frame`；自动布局只排布顶层节点。
* 右键框可「保存为新算子…」；右键框内节点可「移出流程框」；未注册算子 / 缺失依赖的节点或
  占位框可「替换算子…」。保存成功后 `GraphView.operators_changed` 通知主窗口刷新算子库。
* 已连线节点被拖入 / 拖出框时不会自动断开旧连线；新建跨层连线会被层级规则拒绝。

### 自定义算子文件与依赖缺失

* 保存入口：`app/operators/frame.py` 的 `save_frame_operator()`，默认写入
  `operators/custom/`，id 形如 `frame.<slug>`，文件名形如 `frame_<slug>.py`。
* 生成文件继承 `FrameOperator`，模块级定义 `OPERATOR`；`frame_definition` 保存完整内部图。
  `FrameOperator.process()` 固定报错，流程框不能直接当普通算子执行。
* `frame_definition` 的结构为 `format / version / name / inputs / outputs / nodes /
  connections`：
  * 根流程框在定义里用哨兵 `"__frame__"` 表示，边界连线的 `src_node / dst_node`
    写这个 id；
  * 普通节点只保存 `op_id / title / 相对位置 / params / parent_frame`，不内联算子实现；
  * 嵌套手工框保存 `frame`；被引用的其它自定义流程框保存成 `frame_ref`
    （operator_id / name / 端口），避免重复代码；
  * 框内包含原始图像节点时禁止保存。
* 生成文件只导入 `json`、`FrameOperator`、`OperatorSpec / PortSpec`（需要时加
  `DataType`），**不 import 被引用的算子**，因此依赖缺失时仍能被 `OperatorRegistry` 正常加载。
* 拖入画布时 `materialize_frame_operator()` 展开成普通节点 + 流程框 + 连线；缺少普通算子
  的节点显示「失效」，缺少被引用自定义流程框时生成带相同端口的占位框并显示「依赖缺失」，
  用户可通过「替换算子…」换成其它普通算子（可撤销）。替换对话框会排除流程框算子。

### 引擎快照

`ExecutionEngine._capture_graph()` 在快照阶段把每条线解析成真实普通节点端口：普通连线用
`GraphModel.resolve_source()`，流程框的每个输出端口额外记录成
`inputs[(frame_id, "@out:<port_key>")]` 伪输入。这样邻接表、拓扑排序、并行调度与缓存键都只
使用真实节点 id；`_compute_frame()` 只等待各输出端口的真实上游、按 `FramePort` 类型
`coerce_output_value()` 后组装 `NodeResult`。框内普通节点仍作为独立节点执行并回传结果，
错误按分支隔离。详见 [frame_node.md](frame_node.md) 第 5 节。

### 用户自定义算子分类

`app/core/constants.py` 的 `CATEGORY_ORDER` 第一项是 `FRAME_CATEGORY`（「用户自定义算子」），
`OperatorRegistry.tabs()` 按该顺序排序，所以流程框保存生成的算子固定显示在算子库最顶部；
生成文件里的 `OperatorSpec.category` 也固定为 `("用户自定义算子",)`。算子库说明浮层会标注
「流程框算子」，提示拖入画布后会展开为流程框与内部节点。

## 颜色 / dtype 约定

* BGR / BGRA（OpenCV 习惯），灰度 `(H,W)`。
* uint8 / uint16 / int16 / float32 等均可；显示层归一化，状态栏显示原始值。
* `app/core/types.py` 提供 `ImageData`、`to_display_rgb`、`sample_pixel`。

## 时间与缓存语义

| 操作 | 行为 |
| --- | --- |
| 单击节点 | `engine.preview_node(id)`：有同一 revision 快照立即回放；否则让该节点与全部下游失效，后台重算目标节点并递归重算全部下游 |
| 参数拖动 | UI 防抖 120ms；更新模型参数后调用 `preview_node(id)`，实时重算当前节点并递归更新全部下游，不必先点“保存” |
| 保存 | `engine.commit_node(id)`：强制使该节点与下游失效，并重算整棵下游子树（用于确认最终结果） |
| 切换节点 | 与单击节点一致：有有效快照直接显示，否则 `preview_node(id)` 重算该节点与全部下游 |
| 修改连线 / 删除节点 | 连线新增 / 删除 / 替换后立即失效目标节点及全部下游；工作区在 30ms 内合并同一次操作的多条事件，再调用 `engine.update_nodes([...])` 重算，不需要先选中节点。删除节点还会清空引擎快照并取消当前任务 |
| 移动节点 / 自动布局 | 只更新位置，不清空结果快照、不重算；移动流程框时全部后代跟随平移 |
| 连接流程框端口 | 目标框端口按源端口类型绑定并追加新的待定端口；撤销“添加连线”会恢复端口快照 |
| 断开流程框端口 | 只删除连线，不回收已确定的端口类型，也不减少待定端口 |
| 复制粘贴节点 | 新节点与副本间连线一步入栈；连线事件按 30ms 合并后重算新节点及相关下游 |
| 打开工程 | 重新创建引擎并清空缓存 |
| 并行 worker 配置 | `execution/workers`（设置 → 常规 → 后台计算 → 并行线程数）：默认 `min(4, CPU 核数)`，范围 1–16；只影响后续请求，不打断正在运行的任务 |

后台任务由 generation 编号；新请求会取消旧任务，旧结果不得写入界面。一次请求内部
按 DAG 拓扑序把互不依赖的分支并行提交到多线程执行器：兄弟分支真正同时运行，有
依赖的节点等待上游 future；错误按分支隔离，某分支失败不会中断其它分支，上游失败
的下游节点显示“上游节点执行失败”。正在执行的算子不可中断：取消后当前算子会跑完，
但其结果由 generation 丢弃。

## 应用日志

运行日志同时写入 `<应用目录>/config/MineImage.log`（文本）和
`<应用目录>/config/MineImage-logs.db`（SQLite）。SQLite 是日志查看器的数据源：

* `app/core/logstore.py`：`LogStore` 负责建表、追加、筛选、导出与保留最近
  50 000 条；`SQLiteLogHandler` 把 `logging.LogRecord` 落库（异常堆栈进入
  `detail`，也可通过 `extra={"log_detail": "..."}` 附加信息）。每次写入使用
  独立连接并启用 WAL，因此 UI 查询与后台线程写入可以并发。
* `app/main.py`：启动时把 SQLite handler 挂到根 logger，并安装未捕获异常 /
  后台线程异常钩子；异常对话框会显示对应日志 ID。
* `app/ui/log_dialog.py`：非模态查看器，按等级 / 日期 / 模块 / 关键字筛选，
  支持表格复制、详情查看、CSV 导出与清空；`MainWindow.show_logs()` 负责
  复用同一窗口并注入共享的 `LogStore`。

重要失败路径（算子加载、工程打开/保存、图像读取/导出、节点执行、未捕获异常）
都会写入日志；写入失败由 `logging.Handler.handleError` 吞掉，不能阻断业务。
环境变量 `MINEIMAGE_LOG_DB` 可覆盖数据库位置。

## 主要跨模块接口（冻结）

### GraphView（`app/graph/view.py`）

```python
GraphView(workspace: WorkspaceModel, registry: OperatorRegistry,
          undo_stack: QUndoStack, settings: AppSettings, parent=None)

selection_changed = Signal(str)       # "" 表示取消选择
node_activated = Signal(str)          # 双击节点
node_created = Signal(str)            # 画布内新增算子节点
connection_changed = Signal(str)      # 目标节点 id；工作区失效其下游并在 30ms 内合并重算
node_removed = Signal(str)
images_dropped = Signal(list)         # list[str]，拖入的图像路径
status_message = Signal(str)
operators_changed = Signal()          # 保存流程框为新算子后，请求主窗口刷新算子库

set_registry(registry) -> None
notify_result(node_id: str, result: NodeResult) -> None
notify_error(node_id: str, message: str | None) -> None
selected_node_id() -> str | None
set_selected_node(node_id: str) -> None
copy_selected() -> bool              # Ctrl+C：复制选中节点及副本间连线；框会自动带上全部后代
paste_nodes() -> list[str]           # Ctrl+V：粘贴并返回新节点 id，整次一步撤销
auto_layout() -> None                # 只排布顶层节点
fit_graph() -> None
set_thumbnails_enabled(enabled: bool) -> None
add_operator_at(op_id: str, scene_pos: QPointF | None = None) -> str | None
add_frame_at(scene_pos: QPointF | None = None, *, parent_frame: str | None = None,
             title: str = "流程框") -> str
frame_at(scene_pos: QPointF, *, exclude: Iterable[str] = ()) -> str | None
save_frame_as_operator(frame_id: str) -> None
move_out_of_frame(node_id: str) -> None
replace_operator(node_id: str, op_id: str) -> bool
is_operator_registered(op_id: str | None) -> bool
refresh_all() -> None
delete_selected() -> None
```

`NodeResult` 需要 `.primary`、`.preview_image`、`.outputs`、`.summary_text()`、
`.elapsed_ms`、`.revision`；`.outputs` 的 value 可以是任意类型的端口值。

连线校验：GraphView 在创建连线时调用
`app.operators.spec.port_types_compatible(output_port, input_port)`，只允许
`data_type` 相同的端口相连；图像端口还会比较双方 `accepts` 是否有交集。对流程框端口还会
调用 `GraphModel.connection_allowed()` 做层级校验。失败时通过 `status_message` 显示
`connection_error()` 或层级校验生成的中文提示。

### ExecutionEngine（`app/engine/engine.py`）

```python
ExecutionEngine(workspace, registry, settings, parent=None, *, workers=None)

preview_node(node_id) -> None          # 快照命中直接回放；否则递归重算该节点与下游
commit_node(node_id) -> None           # 失效该节点与下游并强制重算整棵子树
refresh_node(node_id) -> None          # 只重算单个节点（上游按需补齐）
update_nodes(node_ids) -> None         # 批量根节点：一次快照、一次请求重算各自子树
invalidate(node_id, *, downstream=True) -> None
invalidate_all() -> None
set_workers(count: int) -> None        # 1..16，只影响后续请求
set_registry(registry) -> None
shutdown() -> None

node_started = Signal(str)
node_finished = Signal(object)         # NodeResult
node_failed = Signal(str, str)         # node id, message
chain_finished = Signal(str, bool)     # root id, success
busy_changed = Signal(bool)
```

`workers` 缺省取 `AppSettings.execution_workers`（`execution/workers`，默认
`min(4, CPU 核数)`，范围 1–16）。同一次请求内多个 worker 并发调用
`Operator.process()`，算子必须线程安全。

流程框节点在抓取快照时被展开成真实上游依赖：框的每个输出端口记录一条
`"@out:<port>"` 伪输入，普通连线用 `GraphModel.resolve_source()` 跨过框边界，
`_compute_frame()` 只等待这些真实来源并按端口类型组装输出。详见
[frame_node.md](frame_node.md) 第 5 节。

### OperatorLibrary（`app/ui/library.py`）

```python
set_registry(registry) -> None
refresh_requested = Signal()
# 拖拽数据 MIME: app.core.constants.MIME_OPERATOR，payload JSON {"id": op_id}
```

一级分类顺序来自 `OperatorRegistry.tabs()`（`CATEGORY_ORDER` 决定），
「用户自定义算子」固定第一；流程框算子在说明浮层中标注「流程框算子」。

### ReplaceOperatorDialog（`app/ui/replace_operator.py`）

```python
ReplaceOperatorDialog(registry: OperatorRegistry | None, *,
                      current_op_id: str | None = None, parent=None)

exec() -> bool                     # 是否确认替换
selected_op_id() -> str | None     # 目标算子 id；取消时为 None
operator_ids() -> list[str]        # 列表中的算子 id（已排除流程框算子）
```

用于画布上失效节点 / 依赖缺失占位框的「替换算子…」。列表排除流程框算子，但保留
`current_op_id` 对应的当前项；搜索覆盖名称 / id / 描述 / 分类 / 关键字。

### PreviewWidget（`app/ui/preview.py`）

```python
set_image(image: ImageData | None, node_title: str = "", info: str = "") -> None
set_original_image(image: ImageData | None, title: str = "原始图像") -> None
set_compare_mode(mode: str) -> None          # single | side_by_side | swipe
set_preview_quality(enabled: bool, max_dim: int) -> None
set_auto_fit_on_open(enabled: bool) -> None  # 首次/尺寸变化时是否 fit
begin_pick(tool: str, count: int = 1, initial=None) -> None
cancel_pick() -> None
fit_to_window() -> None
reset_zoom() -> None
zoom_percent() -> float
image_size() -> tuple[int, int] | None

pixel_info = Signal(object)                  # core.types.PixelSample | None
zoom_changed = Signal(float)
pick_finished = Signal(str, object)          # tool, points/roi/line payload
pick_cancelled = Signal(str)
```

交互拾取数据：

* points: `[[x, y], ...]`（图像坐标，左上为原点）；
* rect: `[x, y, w, h]`；
* line: `[x1, y1, x2, y2]`。

### ParamPanel（`app/ui/params.py`）

```python
load_node(node: NodeModel, operator: Operator) -> None
current_params() -> dict
set_pick_result(key: str, value) -> None

preview_requested = Signal(str, dict)        # node id, params（内部 120ms 防抖）
save_requested = Signal(str, dict)
back_requested = Signal()
pick_requested = Signal(str, str, object)    # node id, param key, tool
```

面板必须用 `app.operators.spec.normalize_params` 做声明式校验，非法时显示错误并
禁用保存。`visible_when/enabled_when` 实时生效。

### SettingsDialog（`app/ui/settings_dialog.py`）

```python
SettingsDialog(settings: AppSettings, parent=None)
applied = Signal()
```

### Theme（`app/ui/theme.py`）

```python
apply_theme(app: QApplication) -> None
```

## 文件写入范围（团队协作）

| 范围 | 负责人 |
| --- | --- |
| `app/core/**`（含 `app/core/frame.py`）, `app/operators/spec.py`, `app/operators/loader.py`, `app/operators/frame.py`, `app/config.py`, `app/ui/main_window.py`, `app/ui/workspace.py`, `app/main.py`, `pyproject.toml`, `tests/test_frame.py`, `tests/test_frame_integration.py` | Lead |
| `app/engine/**`, `app/operators/builtin/**`, `tests/test_engine.py`, `tests/test_operators.py`, `tests/test_frame_engine.py` | engine-ops |
| `app/graph/**`（含 `app/graph/frame_item.py`）, `tests/test_graph.py` | graph-ui |
| `app/ui/preview.py`, `app/ui/library.py`, `app/ui/replace_operator.py`, `app/ui/params.py`, `app/ui/settings_dialog.py`, `app/ui/theme.py`, `tests/test_widgets.py` | shell-ui |
| `docs/**`, `README.md`, `packaging/**`, `examples/**` | qa-docs |

禁止跨范围修改；需要改别人的文件时先给 Lead 发消息。
