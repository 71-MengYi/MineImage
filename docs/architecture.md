# MineImage 架构与模块边界

MineImage 是一个基于 PySide6 的节点式图像处理实验工作台。本文件描述**模块职责与
跨团队接口**，实现细节见各模块 docstring。

## 总体数据流

```
文件/菜单 ──> MainWindow ──> WorkspaceWidget（每个标签页一个）
                                  ├── OperatorLibrary  左侧算子库（搜索/并排一级分类/多级折叠树/拖拽/说明浮层）
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
* **算子**：`app/operators/spec.py` 定义 `OperatorSpec / ParamSpec / PortSpec /
  Operator`；`loader.py` 扫描目录。每个算子文件定义模块级 `OPERATOR`。
  参数联动 `visible_when/enabled_when` 使用受限 `eval`：支持裸参数名与
  `params['key']`，**不支持** `params.key` 属性访问；表达式异常时按 True 处理。
* **引擎**：`app/engine/engine.py`。每次请求抓取图快照，在单线程后台池中执行；
  发出 `node_started/node_finished/node_failed/chain_finished/busy_changed`。
* **画布**：`app/graph/*`，QGraphicsView 自绘。所有图修改走 `QUndoStack`。
  端口改为上输入 / 下输出，连线使用纵向三次贝塞尔曲线（`app/graph/curve.py`）；
  拖拽落点靠近端口时自动连线、落在连线上时自动插入节点，移动与连线命令合并为
  同一个撤销宏。
* **界面**：`app/ui/*`，主窗口与工作区负责组装，其余控件保持可独立测试。

## 颜色 / dtype 约定

* BGR / BGRA（OpenCV 习惯），灰度 `(H,W)`。
* uint8 / uint16 / int16 / float32 等均可；显示层归一化，状态栏显示原始值。
* `app/core/types.py` 提供 `ImageData`、`to_display_rgb`、`sample_pixel`。

## 时间与缓存语义

| 操作 | 行为 |
| --- | --- |
| 单击节点 | `engine.preview_node(id)`：有快照立即显示，否则后台补算上游 |
| 参数拖动 | UI 防抖 120ms；更新模型参数、`invalidate(id, downstream=False)`、`preview_node(id)` |
| 保存 | `engine.commit_node(id)`：使该节点与下游失效，后台重算下游子树 |
| 切换节点 | 直接读取该节点快照；不触发下游计算 |
| 修改连线 / 删除节点 | 由工作区使目标节点及下游失效 |
| 打开工程 | 重新创建引擎并清空缓存 |

后台任务由 generation 编号；新请求会取消旧任务，旧结果不得写入界面。

## 主要跨模块接口（冻结）

### GraphView（`app/graph/view.py`）

```python
GraphView(workspace: WorkspaceModel, registry: OperatorRegistry,
          undo_stack: QUndoStack, settings: AppSettings, parent=None)

selection_changed = Signal(str)       # "" 表示取消选择
node_activated = Signal(str)          # 双击节点
node_created = Signal(str)            # 画布内新增算子节点
connection_changed = Signal(str)      # 目标节点 id；调用方应失效其下游
node_removed = Signal(str)
images_dropped = Signal(list)         # list[str]，拖入的图像路径
status_message = Signal(str)

set_registry(registry) -> None
notify_result(node_id: str, result: NodeResult) -> None
notify_error(node_id: str, message: str | None) -> None
selected_node_id() -> str | None
set_selected_node(node_id: str) -> None
auto_layout() -> None
fit_graph() -> None
set_thumbnails_enabled(enabled: bool) -> None
add_operator_at(op_id: str, scene_pos: QPointF | None = None) -> str | None
refresh_all() -> None
delete_selected() -> None
```

`NodeResult` 需要 `.primary`、`.outputs`、`.elapsed_ms`、`.revision`。

### OperatorLibrary（`app/ui/library.py`）

```python
set_registry(registry) -> None
refresh_requested = Signal()
# 拖拽数据 MIME: app.core.constants.MIME_OPERATOR，payload JSON {"id": op_id}
```

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
| `app/core/**`, `app/operators/spec.py`, `app/operators/loader.py`, `app/config.py`, `app/ui/main_window.py`, `app/ui/workspace.py`, `app/main.py`, `pyproject.toml` | Lead |
| `app/engine/**`, `app/operators/builtin/**`, `tests/test_engine.py`, `tests/test_operators.py` | engine-ops |
| `app/graph/**`, `tests/test_graph.py` | graph-ui |
| `app/ui/preview.py`, `app/ui/library.py`, `app/ui/params.py`, `app/ui/settings_dialog.py`, `app/ui/theme.py`, `tests/test_widgets.py` | shell-ui |
| `docs/**`, `packaging/**`, `examples/**` | qa-docs |

禁止跨范围修改；需要改别人的文件时先给 Lead 发消息。
