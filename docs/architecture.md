# MineImage 架构与模块边界

MineImage 是一个基于 PySide6 的节点式图像处理实验工作台。本文件描述**模块职责与
跨团队接口**，实现细节见各模块 docstring。

## 总体数据流

```
文件/菜单 ──> MainWindow ──> WorkspaceWidget（每个标签页一个）
                                  ├── OperatorLibrary  左侧算子库（搜索/整行折叠一级分类/多级折叠树/拖拽/说明浮层）
                                  ├── GraphView        中间节点画布  ←→ GraphModel
                                  ├── ParamPanel       中间叠层（双击节点进入）
                                  ├── PreviewWidget    右侧预览（缩放/平移/对比/拾取、Imshow/Imagesc/Mesh）
                                  └── ExecutionEngine  后台计算 + 节点快照
                                           ↑
                                  OperatorRegistry（自动扫描 app/operators/builtin + 用户目录）
```

* **模型**：`app/core/model.py`。`WorkspaceModel` 持有 `GraphModel` 与源图像
* **模型**：`app/core/model.py`。`WorkspaceModel` 持有 `GraphModel` 与源图像
  `images: OrderedDict[str, ImageData]`；源节点用 `NodeModel.source_image_ids` 按顺序
  引用其中多张图（`source_ids` 兼容旧的单图 `source_image_id`）。节点参数变化通过
  `NodeModel.touch()` 增加 `revision`。引擎用 revision 丢弃过期结果。
  `WorkspaceModel.display_mode` 保存工作区级预览显示模式（`imshow` / `imagesc` /
  `mesh`），随 `.mi` 工程持久化。
* **矩阵 IO**：`app/core/matrixio.py` 负责 CSV / TSV / TXT 等文本矩阵的解析（分隔符、
  表头 / 行名与缺失值），导入后统一为单通道灰度 `ImageData`；`.npy` 仍按 NumPy 数组
  读取，矩阵数组随源图像以 npz 内嵌保存，工程格式版本不变。
* **端口数据类型**：`app/core/data_types.py` 是端口类型的唯一来源。`DataType`
  定义 `image / number / edges / contours / components` 五种类型；
  `NumberValue / EdgesData / ContoursData / ComponentsData` 是结构化值的运行时
  容器；图像端口的值可以是单张 `ImageData`，也可以是多图容器 `ImageList`（图像
  矩阵，仍属于 `DataType.IMAGE`）。`coerce_data_type()` 归一化中英文别名，
  `render_preview()` / `summarize_value()` 供算子显式可视化与摘要使用。`PortSpec.data_type` 声明端口类型
  （默认图像），`port_types_compatible() / connection_error()` 决定连线是否允许，
  `validate_inputs()` 在执行前校验输入，`output_to_dict() / coerce_output_value()`
  按端口类型规范化算子返回值。
* **类型化数据流**：数据流从“图像容器”升级为“类型化端口值”。连线传递的不再只
  是 `ImageData` / `ImageList`，也可能是 `NumberValue`、`EdgesData`、`ContoursData`
  或 `ComponentsData`；`NodeResult.outputs` 是泛化的 `dict[str, Any]`，
  `preview_image` 取第一个图像输出的首张、没有图像输出时为 `None`，`summary_text()`
  汇总各端口摘要。
* **结构化数据不再自动预览**：`NodeResult.preview_image` 只返回图像输出；数值、
  边缘、轮廓、连通域不再由 `render_preview()` 转成主预览或节点缩略图。节点卡片对
  单个数值直接显示、对图像显示缩略图、对数组提示「数组数据 / 双击查看」；参数面板的
  「输出数据」区在结果包含非图像输出时自动出现，以文本详情展示数组的数量、画布尺寸、
  形状、前几个点等，不再生成预览图。
* **万能端口与输出节点**：`PortSpec(any_type=True)` 表示万能端口，输入可接收五类
  数据中的任意一种，输出按运行时类型原样透传；连线校验只要任意一端为万能端口即兼容。
  `其他 → 可视化 → 输出预览`（`output.display`）就用一个 `data` 输入和一个
  `data` 输出实现任意链路的终点。
* **便签节点**：`其他 → 可视化 → 便签`（`note.sticky`）同样使用万能入口 / 出口，
  但节点卡片由 `OperatorSpec.canvas_text_param` 声明为文字卡片：只显示可编辑的便签
  文字（暖色样式、按行数自适应高度），不渲染缩略图 / 数值 / 数组提示。
  `OperatorSpec.allow_empty_output` 允许它在未接输入时返回空输出而不算失败，
  输入端口声明为 `optional=True`，因此便签可以单独放在画布上。
* **算子**：`app/operators/spec.py` 定义 `OperatorSpec / ParamSpec / PortSpec /
  Operator`；`loader.py` 扫描目录。每个算子文件定义模块级 `OPERATOR`。
  参数联动 `visible_when/enabled_when` 使用受限 `eval`：支持裸参数名与
  `params['key']`，**不支持** `params.key` 属性访问；表达式异常时按 True 处理。
* **引擎**：`app/engine/engine.py`。每次请求在主线程抓取一份不可变的图快照，再在
  后台线程池中按 DAG 拓扑序提交节点任务：兄弟分支并行执行，有依赖的节点等待上游
  future 完成；拓扑排序用 FIFO Kahn 展开，按“到源节点的层数”逐层向下（广度优先）。
  `update_nodes` / `preview_node` / `commit_node` 递归重算目标节点与全部下游，
  `refresh_node` 只重算单个节点；`precompute_all` 在打开工程后以全部节点为根、一次
  算完整张图（含未接线节点与算子组容器），`chain_finished` 只按末端节点报告。结果
  按 `generation + revision` 双重校验丢弃过期写回，节点错误按分支隔离。算子
  `process()` 会被多个线程并发调用，
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
| `IMAGE` | `image` | `ImageData` / `ImageList` | 默认端口类型；单张与图像矩阵共用同一类型；`ndarray` 自动包装并按形状推断颜色空间 |
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

## 图像矩阵与批量执行

图像矩阵（`ImageList`）让一次打开的多张图在同一个源节点里流转，也让算子按图批量执行。
它不新增端口类型：对连线校验、工程格式和界面来说仍是 `image` 类型。

### 数据层

* **容器**：`app/core/types.py` 的
  `ImageList(images: tuple[ImageData, ...], name="图像矩阵")` 是不可变容器，提供
  `count / first / image_at() / summary() / copy()`；`image_values()` 把单张 / 矩阵
  统一展开成 `tuple[ImageData, ...]`，`first_image()` 取首张。
* **类型**：`data_type_of(ImageList)` 返回 `DataType.IMAGE`；`validate_inputs()` 对图像
  端口逐张校验颜色空间，空矩阵直接报错；`coerce_output_value()` 接受 `ImageList`，
  也把非空 `list/tuple[ImageData]` 包装成矩阵。矩阵内各图尺寸 / 颜色空间 / dtype
  可以不同，不会自动缩放或统一。
* **节点模型**：`NodeModel.source_image_ids` 保存源节点引用的全部图像 id 与顺序；
  `source_ids` 属性在列表为空时回退旧工程的 `source_image_id`。`to_dict()` 写出
  `source_image_ids`，`from_dict()` 把 `source_image_id` 合并进列表并去重。
* **工作区**：`WorkspaceModel.add_source_images(images, paths, ...)` 为每张图分配
  `img` id 并创建一个源节点；`add_source_image()` 委托给前者处理单图。
  `images_for_node(node)` 按 `source_ids` 顺序返回全部 `ImageData`，
  `image_for_node()` 只返回第一张（兼容单图调用方）；`prune_unused_images()` 保留
  所有被 `source_ids` 引用的图。
* **工程格式**：`.mi` 的 `project.json` 记录源节点的 `source_image_ids`，图像仍逐张
  存为 `images/<image_id>.npz`；保存 / 打开不改变单图工程的文件结构，多图工程只是
  多几个 npz。`load_project()` 会校验每个源节点引用的图像都存在。
* **界面创建入口**：`WorkspaceWidget.add_image_matrix(paths)` 用于“文件 → 打开图像”
  多选：读取失败的文件跳过并汇总提示，成功的图放进同一个源节点；
  `add_image(path)` 仍是一张图一个源节点，“添加图像”多选时逐文件调用它。

### 引擎批量语义

* **源节点输出**：`_compute_source()` 按 `source_image_ids` 取图；1 张输出
  `ImageData`，多张输出 `ImageList(name=节点标题)`，图片缺失时报
  「源图像数据缺失，请重新载入图像」。
* **结果便捷属性**：`NodeResult.primary_images / has_image_matrix / image_count /
  image_at()` 直接基于第一个图像输出（`ImageData` 或 `ImageList`）；`primary /
  preview_image` 仍取首张，保证旧的单图调用方不变。
* **批量识别**：`_batch_plan()` 把 `ImageList` 视为批量输入，也把同构的
  `list/tuple`（`data_type_of` 能识别，例如逐张数值结果）视为批量输入。所有批量
  输入中最大长度为 N；每个批量输入长度必须为 1 或 N，长度为 1 的输入广播到每
  一张；长度不在 `{1, N}` 时节点失败，提示「输入图像矩阵数量不一致（...）」。
  空批量输入单独报错。只有 1 张的矩阵按单张解包，保持原有单图执行路径与耗时。
* **逐张执行**：`_process_batch()` 对 i=0..N-1 构造一份输入（批量端口取第
  `min(i, len-1)` 项，其余输入原样），逐张调用 `validate_inputs / validate /
  process / output_to_dict`；参数只在循环外规范化一次。
* **输出聚合**：每个输出端口按声明类型聚合：图像端口（以及实际逐项都是图像的万能
  端口）合并为 `ImageList`（逐张结果按顺序展开）；数值 / 边缘 / 轮廓 / 连通域
  等其他端口保持逐张 `list` 顺序。某张缺少某个输出端口时节点失败。任一张执行失败
  时整个节点失败，错误信息包含「第 i+1 张」。
* **算子组**：组内节点仍作为普通节点独立执行；组输出端口的来源若是图像矩阵，按端口
  类型 `coerce_output_value()` 保留 / 包装矩阵，详见 [group_node.md](group_node.md)
  第 5 节。

### 画布与预览

* **NodeItem 网格**：`app/graph/node_item.py` 在结果第一个图像输出为多图时进入
  `_content_mode="grid"`，`_grid_images` 保存全部图像，节点中部按方形单元格排布；
  `grid_columns()` 的列数规则为 1 张 1 列、2–4 张 2 列、5 张及以上 3 列（1 张实际
  仍走单图缩略图模式），单元格边长不超过缩略图设置，节点高度随行数增长。每格绘制
  缩略图、左上角序号与选中描边。
* **命中测试**：`NodeItem.image_at(index)`、`image_tile_rects()` 暴露网格数据，
  `image_index_at(QPointF)` 用单元格矩形返回序号（未命中 / 非网格返回 `None`）。
* **选中序号**：`_selected_image_index` 默认 0，`set_result()` 后按新张数 clamp，
  张数变少时尽量保留原索引；`set_selected_image_index(index)` 同样 clamp。
* **点击语义**：`NodeItem.mousePressEvent` 先做 `image_index_at()`；命中时
  `_select_grid_image()` 只更新序号并经 `GraphView.on_node_image_selected()` 发射
  `node_image_selected(str, int)`，`accept()` 后直接返回，不经过 `QGraphicsItem`
  的选中 / 拖动逻辑。因此点缩略图不会选中节点、不会移动节点；双击缩略图也不会
  打开参数面板。点击标题 / footer 等其它区域才交给基类选中节点。
* **GraphView 接口**：`node_image_selected = Signal(str, int)`（节点 id、0 起序号）、
  `on_node_image_selected(item, index)`、`selected_image_index(node_id)`、
  `set_selected_image_index(node_id, index)`；`_recreate_node_item()` 重建条目时
  保留旧选中序号。
* **缩略图关闭**：`set_thumbnails_enabled(False)` 后不建网格，`_content_mode="matrix"`，
  只显示「图像矩阵 ×N」文字；界面此时不提供点选单图的入口。
* **工作区预览**：`WorkspaceWidget._result_images()` 优先用
  `NodeResult.primary_images` 展开矩阵；`_show_result()` 用节点当前选中序号取一张，
  状态栏详情显示「图像矩阵 N 张 · 第 i/N 张 · 色彩空间 · dtype」。
  `_refresh_original()` 在结果节点上用结果节点的选中序号取源矩阵中的同一张（越界
  回退第一张），导出当前节点图像也取这一张。

## 算子组节点与自定义算子

算子组（Group Node）是画布上的容器节点：自身带动态数量的输入 / 输出端口，内部可以放普通
算子节点，也可以继续嵌套算子组；整个算子组可保存成「用户自定义算子」，再从算子库拖回画布复用。
实现说明见 [group_node.md](group_node.md)，用户操作见 [user_guide.md](user_guide.md) 第 8 节，
算子文件写法见 [operator_guide.md](operator_guide.md) 第 12 节。

### 数据模型

* **扁平图 + parent_group 链**：组内节点仍在同一个 `GraphModel.nodes` 里，通过
  `NodeModel.parent_group` 指向所属算子组；嵌套算子组也只是多一级链。`GraphModel` 提供
  `direct_children / group_subtree / is_inside / node_container` 查询，不引入第二套图结构。
* **NodeModel.group**：只有算子组节点（`op_id == GROUP_OP_ID`，即 `"__group__"`）使用，
  `is_group` 为 `op_id == GROUP_OP_ID and group is not None`。
* **GroupModel / GroupPort**（`app/core/group.py`）：组的输入、输出端口列表与来源
  `operator_id`。每侧始终保留一个待定端口（`data_type=None`，对应
  `PortSpec(any_type=True)`）；连线确定类型后 `GroupModel.bind()` 把待定端口定类型并
  `ensure_pending()` 追加新的待定端口。删除连线时 `GraphModel` 会回收两侧都已空闲的
  已定类型端口，空闲待定端口顶替其 key，保证端口数量按连接动态变化。
* **工程序列化**：`NodeModel.to_dict()` 写出 `parent_group`（非空时）与 `group`；
  `GraphModel.from_dict()` 加载后修复悬空父算子组（父节点不存在 / 不是算子组时回退顶层）。

### 端口层级与内外侧

算子组的每个端口在画布上同时有外侧与内侧两个圆形连接点（内侧向内偏移 14px，图形相同）。
模型用 `GraphModel.port_level(node_id, port_key, role)` 定义层级，只有层级相同才允许连线：

| 端口 | 作为源（`src`） | 作为目标（`dst`） |
| --- | --- | --- |
| 普通节点端口 | 节点的 `parent_group` | 节点的 `parent_group` |
| 算子组输入端口 | 算子组 id（内侧，组内数据源） | 组的 `parent_group`（外侧，接收组外数据） |
| 算子组输出端口 | 组的 `parent_group`（外侧，向组外供数） | 算子组 id（内侧，接收组内数据） |

* `connection_role()` 把连线归类为 `normal / group_input_inner / group_output_inner /
  group_input_outer / group_output_outer / invalid`；
* `connection_allowed()` 在节点、端口存在性之外做层级校验，失败时返回中文原因；
* 回环检测 `effective_successors()` 忽略 `group_input_inner` 边界边，避免把
  「组输入 → 组内节点 → 组输出」误判为环；
* `resolve_source(src_node, src_port)` 递归跨过组边界，解析出真正的普通节点端口：
  组输入端口沿外侧入边解析，组输出端口沿内侧入边解析，断开时返回 `None`；
  `group_output_sources(group_id)` 返回每个输出端口的真实来源，供引擎使用。

### 画布与交互

* `app/graph/group_item.py` 的 `GroupItem` 绘制圆角容器、标题栏与「失效 / 依赖缺失」徽标；
  外框输入端口在顶边、输出端口在底边，内侧端口在组内 14px 处。
* 算子组几何不单独保存：`GraphView._sync_group_geometries()` 按**直接子节点**包围盒 + padding
  反推算子组的位置与尺寸（无子节点默认 420×220）；嵌套算子组按深度由内向外同步。
* 空白处右键「添加算子组」；从算子库拖入算子组区域时取最深命中的嵌套算子组作为 `parent_group`；
  拖动普通节点按中心命中的算子组更新所属关系；拖动算子组整体平移全部后代。
* 拖动已连线节点进入 / 移出算子组时，`ReconnectGroupMoveCommand` 会把跨层连线改接到
  组边界端口上（按公共祖先补齐多层边界），并把不需要的旧连线与端口一并回滚；整次拖动一步撤销。
* 删除算子组会连同全部后代与相关连线一步撤销；复制 / 粘贴算子组会带上后代与内部连线并重映射
  `parent_group`。自动布局先排布顶层，再按深度从外到内排布每个算子组的直接子节点。
* 右键算子组可「保存为新算子…」或「解散算子组」；`DissolveGroupCommand` 会把边界端口的
  内外连线两两改接成直连，保留内部节点与位置，整次操作一步撤销。右键算子组内节点可
  「移出算子组」；未注册算子 / 缺失依赖的节点或占位算子组可「替换算子…」。
  保存成功后 `GraphView.operators_changed` 通知主窗口刷新算子库。
* 已连线节点被拖入 / 拖出算子组时不再保留跨层直连；新建跨层连线仍会被层级规则拒绝。

### 自定义算子文件与依赖缺失

* 保存入口：`app/operators/group.py` 的 `save_group_operator()`，默认写入
  `operators/custom/`，id 形如 `group.<slug>`，文件名形如 `group_<slug>.py`。
* 生成文件继承 `GroupOperator`，模块级定义 `OPERATOR`；`group_definition` 保存完整内部图。
  `GroupOperator.process()` 固定报错，算子组不能直接当普通算子执行。
* `group_definition` 的结构为 `format / version / name / inputs / outputs / nodes /
  connections`：
  * 根算子组在定义里用哨兵 `"__group__"` 表示，边界连线的 `src_node / dst_node`
    写这个 id；
  * 普通节点只保存 `op_id / title / 相对位置 / params / parent_group`，不内联算子实现；
  * 嵌套手工算子组保存 `group`；被引用的其它自定义算子组保存成 `group_ref`
    （operator_id / name / 端口），避免重复代码；
  * 组内包含原始图像节点时禁止保存。
* 生成文件只导入 `json`、`GroupOperator`、`OperatorSpec / PortSpec`（需要时加
  `DataType`），**不 import 被引用的算子**，因此依赖缺失时仍能被 `OperatorRegistry` 正常加载。
* 拖入画布时 `materialize_group_operator()` 展开成普通节点 + 算子组 + 连线；缺少普通算子
  的节点显示「失效」，缺少被引用自定义算子组时生成带相同端口的占位算子组并显示「依赖缺失」，
  用户可通过「替换算子…」换成其它普通算子（可撤销）。替换对话框会排除算子组。

### 引擎快照

`ExecutionEngine._capture_graph()` 在快照阶段把每条线解析成真实普通节点端口：普通连线用
`GraphModel.resolve_source()`，算子组的每个输出端口额外记录成
`inputs[(group_id, "@out:<port_key>")]` 伪输入。这样邻接表、拓扑排序、并行调度与缓存键都只
使用真实节点 id；`_compute_group()` 只等待各输出端口的真实上游、按 `GroupPort` 类型
`coerce_output_value()` 后组装 `NodeResult`。组内普通节点仍作为独立节点执行并回传结果，
错误按分支隔离。详见 [group_node.md](group_node.md) 第 5 节。

### 用户自定义算子分类

`app/core/constants.py` 的 `CATEGORY_ORDER` 第一项是 `CUSTOM_OPERATOR_CATEGORY`（「用户自定义算子」），
`OperatorRegistry.tabs()` 按该顺序排序，所以算子组保存生成的算子固定显示在算子库最顶部；
生成文件里的 `OperatorSpec.category` 也固定为 `("用户自定义算子",)`。算子库说明浮层会标注
「算子组」，提示拖入画布后会展开为算子组与内部节点。

## 颜色 / dtype 约定

* BGR / BGRA（OpenCV 习惯），灰度 `(H,W)`。
* uint8 / uint16 / int16 / float32 等均可；显示层归一化，状态栏显示原始值。
* `app/core/types.py` 提供 `ImageData`、`ImageList`、`to_display_rgb`、`sample_pixel`；
  图像矩阵的 `color_space` / `dtype_name` 在成员不一致时返回「混合」。

## 时间与缓存语义

| 操作 | 行为 |
| --- | --- |
| 单击节点 | `engine.preview_node(id)`：有同一 revision 快照立即回放；否则让该节点与全部下游失效，后台重算目标节点并递归重算全部下游 |
| 单击矩阵缩略图 | 只切换当前显示的单图序号并刷新右侧预览；不改变节点选择、不触发重算，使用已有快照 |
| 参数拖动 | UI 防抖 120ms；更新模型参数后调用 `preview_node(id)`，实时重算当前节点并递归更新全部下游，不必先点“保存” |
| 保存 | `engine.commit_node(id)`：强制使该节点与下游失效，并重算整棵下游子树（用于确认最终结果） |
| 切换节点 | 与单击节点一致：有有效快照直接显示，否则 `preview_node(id)` 重算该节点与全部下游 |
| 修改连线 / 删除节点 | 连线新增 / 删除 / 替换后立即失效目标节点及全部下游；工作区在 30ms 内合并同一次操作的多条事件，再调用 `engine.update_nodes([...])` 重算，不需要先选中节点。删除节点还会清空引擎快照并取消当前任务 |
| 移动节点 / 自动布局 | 只更新位置，不清空结果快照、不重算；移动算子组时全部后代跟随平移 |
| 连接算子组端口 | 目标算子组端口按源端口类型绑定并追加新的待定端口；若端口是空闲待定端口则直接复用，撤销“添加连线”会恢复端口快照 |
| 断开算子组端口 | 若端口内外两侧都已无连线，回收已定类型端口并让空闲待定端口顶替其 key；撤销删除连线会恢复端口快照 |
| 拖动节点进 / 出算子组 | 跨层连线自动改接组边界端口：优先复用空闲端口（待定端口 / 旧版本遗留的已定类型空闲端口），并按节点输入 / 输出声明顺序分配端口；整次拖动一步撤销 |
| 复制粘贴节点 | 新节点与副本间连线一步入栈；连线事件按 30ms 合并后重算新节点及相关下游 |
| 打开工程 | 重新创建引擎并清空缓存；随后 `precompute_all()` 以全部节点为根、按广度优先层序一次算完每个子树（含未接线节点与算子组容器），结果与缩略图不再需要手动点击触发；`chain_finished` 只对每个末端节点报告一次 |
| 并行 worker 配置 | `execution/workers`（设置 → 常规 → 后台计算 → 并行线程数）：默认 `min(4, CPU 核数)`，范围 1–16；只影响后续请求，不打断正在运行的任务 |
| 算子文件变化 | 目录监视默认开启（每 3 秒比较 mtime/size）；关闭监视时，每次计算请求前由引擎比较 `file_snapshot()`，变化则重建注册表后再抓快照，本次请求即使用新代码 |

后台任务由 generation 编号；新请求会取消旧任务，旧结果不得写入界面。一次请求内部
按 DAG 拓扑序把互不依赖的分支并行提交到多线程执行器：兄弟分支真正同时运行，有
依赖的节点等待上游 future；错误按分支隔离，某分支失败不会中断其它分支，上游失败
的下游节点显示“上游节点执行失败”。拓扑排序使用 FIFO Kahn，出队顺序按到源节点的
层数分层，即广度优先；打开工程时的整图预计算同样按该层序提交。正在执行的算子不可
中断：取消后当前算子会跑完，但其结果由 generation 丢弃。

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
node_image_selected = Signal(str, int)  # 节点 id, 图像矩阵中的序号（0 起）
node_activated = Signal(str)          # 双击节点
node_created = Signal(str)            # 画布内新增算子节点
connection_changed = Signal(str)      # 目标节点 id；工作区失效其下游并在 30ms 内合并重算
node_removed = Signal(str)
images_dropped = Signal(list)         # list[str]，拖入的图像路径
status_message = Signal(str)
operators_changed = Signal()          # 保存算子组为新算子后，请求主窗口刷新算子库

set_registry(registry) -> None
notify_started(node_id: str) -> None  # 节点开始执行：卡片右下角实时计时
notify_result(node_id: str, result: NodeResult) -> None
notify_error(node_id: str, message: str | None) -> None
clear_running() -> None              # 新请求取消旧请求时清掉遗留的运行中计时
selected_node_id() -> str | None
set_selected_node(node_id: str) -> None
selected_image_index(node_id: str) -> int     # 节点卡片当前选中的图像序号
set_selected_image_index(node_id: str, index: int) -> None
copy_selected() -> bool              # Ctrl+C：复制选中节点及副本间连线；算子组会自动带上全部后代
paste_nodes() -> list[str]           # Ctrl+V：粘贴并返回新节点 id，整次一步撤销
auto_layout() -> None                # 顶层 + 各层算子组内部节点一起布局
fit_graph() -> None
set_thumbnails_enabled(enabled: bool) -> None
add_operator_at(op_id: str, scene_pos: QPointF | None = None) -> str | None
add_group_at(scene_pos: QPointF | None = None, *, parent_group: str | None = None,
             title: str = "算子组") -> str
group_at(scene_pos: QPointF, *, exclude: Iterable[str] = ()) -> str | None
save_group_as_operator(group_id: str) -> None
dissolve_group(group_id: str) -> None
move_out_of_group(node_id: str) -> None
replace_operator(node_id: str, op_id: str) -> bool
is_operator_registered(op_id: str | None) -> bool
refresh_all() -> None
delete_selected() -> None
```

`NodeResult` 需要 `.primary`、`.preview_image`、`.outputs`、`.summary_text()`、
`.elapsed_ms`、`.revision`；`.outputs` 的 value 可以是任意类型的端口值。

节点卡片右下角的耗时由 GraphView 维护：`notify_started()` 开始按 100ms tick 显示
实时耗时；`notify_result()` / `notify_error()` 结束计时，成功时播放 900ms 的绿色渐隐
动画后恢复常规灰色（失败直接显示「错误」）。状态栏「耗时」由 WorkspaceWidget 计时，
通过 `status_info["elapsed_state"]` 传递 idle / running / done，供主窗口做同款完成动画。

连线校验：GraphView 在创建连线时调用
`app.operators.spec.port_types_compatible(output_port, input_port)`，只允许
`data_type` 相同的端口相连；图像端口还会比较双方 `accepts` 是否有交集。对算子组端口还会
调用 `GraphModel.connection_allowed()` 做层级校验。失败时通过 `status_message` 显示
`connection_error()` 或层级校验生成的中文提示。

### ExecutionEngine（`app/engine/engine.py`）

```python
ExecutionEngine(workspace, registry, settings, parent=None, *, workers=None)

preview_node(node_id) -> None          # 快照命中直接回放；否则递归重算该节点与下游
commit_node(node_id) -> None           # 失效该节点与下游并强制重算整棵子树
refresh_node(node_id) -> None          # 只重算单个节点（上游按需补齐）
update_nodes(node_ids) -> None         # 批量根节点：一次快照、一次请求重算各自子树
precompute_all() -> None               # 打开工程后：作废旧快照，按广度优先算完整张图
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
request_started = Signal(int)          # 新请求真正提交时的 generation；旧请求残留的“运行中”状态据此清理
operator_reload_requested = Signal()   # 运行前发现用户算子文件变化，请求上层重建注册表
```

`workers` 缺省取 `AppSettings.execution_workers`（`execution/workers`，默认
`min(4, CPU 核数)`，范围 1–16）。同一次请求内多个 worker 并发调用
`Operator.process()`，算子必须线程安全。

所有对外计算入口（`preview_node` / `commit_node` / `refresh_node` /
`update_nodes` / `precompute_all`）都会在缓存判断与图快照之前调用
`_maybe_reload_operator_files()`：比较 `OperatorRegistry.file_snapshot()` 的
mtime/size，发现用户算子文件变化时同步发出 `operator_reload_requested`。
主窗口收到后以 `refresh_operators(preview=False)` 重建注册表，当前请求继续使用
新注册表执行；`_reloading_operators` 标志防止刷新期间重入产生第二个请求。

算子组节点在抓取快照时被展开成真实上游依赖：组的每个输出端口记录一条
`"@out:<port>"` 伪输入，普通连线用 `GraphModel.resolve_source()` 跨过组边界，
`_compute_group()` 只等待这些真实来源并按端口类型组装输出。详见
[group_node.md](group_node.md) 第 5 节。

### OperatorLibrary（`app/ui/library.py`）

```python
OperatorLibrary(parent=None, *, settings: AppSettings | None = None,
                user_operator_dir: Path | str | None = None)

set_registry(registry) -> None
refresh_requested = Signal()          # 新建 / 改名 / 删除后请求上层重扫
status_message = Signal(str)          # 文件操作失败提示（主窗口转状态栏）

# 右键菜单与可测试入口
build_context_menu(item) -> QMenu
create_operator(category=None) -> str | None
create_category_named(parent, name) -> Path | None
rename_operator_to(op_id, new_stem) -> bool
delete_operator(op_id, confirm=True) -> bool
open_editor(op_id) -> bool
select_operator(op_id) -> bool
begin_rename_operator(op_id) -> bool
# 拖拽数据 MIME: app.core.constants.MIME_OPERATOR，payload JSON {"id": op_id}
```

一级分类顺序来自 `OperatorRegistry.tabs()`（`CATEGORY_ORDER` 决定），并把
`known_categories()` 里的 0 算子空分类合并进树；「用户自定义算子」固定第一；
算子组在说明浮层中标注「算子组」。非内置条目文本使用
`OperatorEntry.display_name`：界面新建（带 `MINEIMAGE_UI_CREATED` 标记）的算子
跟随文件 stem，其余仍显示 `spec.name`；原地改名只改文件名（必要时同步模板
`name`），不改变算子 id。

### ReplaceOperatorDialog（`app/ui/replace_operator.py`）

```python
ReplaceOperatorDialog(registry: OperatorRegistry | None, *,
                      current_op_id: str | None = None, parent=None)

exec() -> bool                     # 是否确认替换
selected_op_id() -> str | None     # 目标算子 id；取消时为 None
operator_ids() -> list[str]        # 列表中的算子 id（已排除算子组）
```

用于画布上失效节点 / 依赖缺失占位算子组的「替换算子…」。列表排除算子组，但保留
`current_op_id` 对应的当前项；搜索覆盖名称 / id / 描述 / 分类 / 关键字。

### PreviewWidget（`app/ui/preview.py`）

```python
set_image(image: ImageData | None, node_title: str = "", info: str = "") -> None
set_original_image(image: ImageData | None, title: str = "原始图像") -> None
set_compare_mode(mode: str) -> None          # single | side_by_side | swipe
set_display_mode(mode: str) -> None          # imshow | imagesc | mesh
display_mode() -> str
mesh_camera() -> tuple[float, float]          # 方位角, 仰角（度）
reset_mesh_view() -> None                     # 恢复默认视角并 fit
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

显示模式与对比布局相互独立：`imshow` 按原始颜色 / 灰度显示；`imagesc` 用
`app.core.types.to_scaled_rgb()` 把数据 min/max 拉伸后套用 parula 色带；`mesh`
用 `app.core.types.intensity_grid()` 把强度降采样（最长边 320）成高度场，按
方位角 / 仰角投影成平滑着色曲面（不绘制网格线）。

* mesh 优先用**离屏 OpenGL 3.3** 渲染（2× 超采样 + 4× MSAA FBO，MSAA 读回后
  缩放回面板）；上下文不可用时自动回退到软件条带渲染（降采样到 160 边）。
* 交互：左键拖动旋转（仰角不限制，可连续拖过顶部 / 底部）、滚轮直接缩放、
  `Shift+左键` / 右键拖动平移、双击恢复默认视角并适应窗口；二维模式仍是
  `Ctrl+滚轮` 缩放、左键平移。
* mesh 的投影坐标仍落在图像宽高组成的虚拟框内，因此复用同一套
  `fit_to_window` / 缩放 / 平移逻辑；像素采样与交互拾取在 mesh 模式下不适用
  （采样返回 `None`，`begin_pick` 抛 `ValueError`，由工作区提示用户切回前两种模式）。

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

「算子目录」页包含用户算子目录、内置算子开关、目录监视开关、默认编辑器选择
（`operators/editor`）与配置文件路径；默认编辑器由 `app/core/editor.py` 探测
VS Code / PyCharm 等，亦可用 `system` 或 `path:<可执行文件>` 指定。

### Theme（`app/ui/theme.py`）

```python
apply_theme(app: QApplication) -> None
```

## 文件写入范围（团队协作）

| 范围 | 负责人 |
| --- | --- |
| `app/core/**`（含 `app/core/group.py`、`app/core/editor.py`、`app/core/operator_files.py`）, `app/operators/spec.py`, `app/operators/loader.py`, `app/operators/template.py`, `app/operators/group.py`, `app/config.py`, `app/ui/main_window.py`, `app/ui/workspace.py`, `app/main.py`, `pyproject.toml`, `tests/test_group.py`, `tests/test_group_integration.py`, `tests/test_image_matrix_core.py`, `tests/test_image_matrix_ui.py` | Lead |
| `app/engine/**`, `app/operators/builtin/**`, `tests/test_engine.py`, `tests/test_operators.py`, `tests/test_group_engine.py`, `tests/test_image_matrix_engine.py` | engine-ops |
| `app/graph/**`（含 `app/graph/group_item.py`）, `tests/test_graph.py`, `tests/test_image_matrix_grid.py` | graph-ui |
| `app/ui/preview.py`, `app/ui/display_modes.py`, `app/ui/library.py`, `app/ui/replace_operator.py`, `app/ui/params.py`, `app/ui/settings_dialog.py`, `app/ui/theme.py`, `tests/test_widgets.py` | shell-ui |
| `docs/**`, `README.md`, `packaging/**`, `examples/**` | qa-docs |

禁止跨范围修改；需要改别人的文件时先给 Lead 发消息。
