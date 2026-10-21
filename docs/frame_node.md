# 流程框节点（Frame Node）实现说明

流程框是画布上的容器节点：它自身带动态数量的输入 / 输出端口，内部可以放普通算子节点，
也可以继续嵌套流程框；整个框可以保存成「用户自定义算子」，之后从算子库拖回画布复用。

本文面向修改 MineImage 的开发者，说明流程框的数据模型、序列化格式、画布交互与引擎执行方式。
用户操作说明见 [user_guide.md](user_guide.md) 第 8 节；从算子作者视角看
`FrameOperator` 的写法见 [operator_guide.md](operator_guide.md) 第 12 节。

## 1. 概念与常量

| 名称 | 位置 | 说明 |
| --- | --- | --- |
| `FRAME_OP_ID` | `app/core/constants.py` | 流程框节点的伪算子 id：`"__frame__"` |
| `FRAME_BOUNDARY_ID` | `app/core/frame.py` | 定义文件里代表“流程框自身边界”的哨兵 id，同样是 `"__frame__"` |
| `FRAME_CATEGORY` | `app/core/constants.py` | 流程框生成的自定义算子所属一级分类：`"用户自定义算子"` |
| `FRAME_DEFINITION_FORMAT` | `app/core/frame.py` | `"mineimage-frame-definition"` |
| `FRAME_DEFINITION_VERSION` | `app/core/frame.py` | 当前定义格式版本：`1` |
| `INVALID_LEVEL` | `app/core/frame.py` | `GraphModel.port_level` 对非法端口的返回值：`"__invalid__"` |
| `FRAME_OUTPUT_INPUT_PREFIX` | `app/engine/engine.py` | 引擎快照里框输出伪输入的 key 前缀：`"@out:"` |

三个容易混淆的概念：

* **端口本身**：`FrameModel.inputs / outputs` 中的 `FramePort`，保存 key、显示名与数据类型；
  数据类型为 `None` 时是「待定（未知类型）」。
* **外侧端口 / 内侧端口**：画布上同一个连接点绘制成两个圆形端口。外侧端口负责框外连线，
  内侧端口负责框内连线；两者共享同一个 `FramePort`，图形相同，内侧端口向内偏移 14px。
* **端口层级**：判断两条线能否相连的“作用域”。普通节点、框输入端口、框输出端口在作为
  源或目标时属于不同的层级，详见 2.4。

## 2. 数据模型

### 2.1 `FramePort`

```python
@dataclass
class FramePort:
    key: str                    # "input_1" / "output_1"
    label: str                  # "输入 1" / "输出 1"
    data_type: DataType | None  # None = 待定（任意类型）
```

* `pending`：`data_type is None`；`type_text` 返回「任意类型（待定）」或类型中文名。
* `port_spec()`：待定端口返回 `PortSpec(..., any_type=True)`，已定类型返回普通
  `PortSpec(..., data_type=...)`。因此待定端口可以接收任意已知类型。
* `to_dict()` / `from_dict()`：数据类型写成 `"image"`、`"number"` 等值；无法识别的类型
  在加载时按「待定」处理，而不是报错。
* 端口 key 由 `FrameModel` 生成：输入为 `input_<N>`，输出为 `output_<N>`；显示名分别为
  「输入 N」「输出 N」。

### 2.2 `FrameModel`

```python
@dataclass
class FrameModel:
    inputs: list[FramePort]
    outputs: list[FramePort]
    operator_id: str | None   # 来自哪个自定义算子文件；手工创建的框为 None
```

关键方法：

| 方法 | 行为 |
| --- | --- |
| `create_default(operator_id=None)` | 新建框的初始状态：输入、输出各一个待定端口 |
| `ensure_pending(is_input)` | 该侧没有待定端口时追加一个；已有则直接返回 |
| `ensure_pending_ports()` | 输入、输出两侧各确保一个待定端口 |
| `bind(key, is_input, data_type)` | 把待定端口绑定为具体类型；成功时自动追加新的待定端口并返回 `True` |
| `bind_from_spec(key, is_input, spec)` | 从 `PortSpec` 绑定；万能端口、无类型声明时不绑定 |
| `port_spec(key, is_input)` | 查询端口的 `PortSpec` |
| `spec(title)` | 生成 `id=FRAME_OP_ID`、分类为 `("流程框",)` 的动态 `OperatorSpec`，供节点渲染与连线校验 |
| `copy()` / `to_dict()` / `from_dict()` | 深拷贝与序列化；`from_dict` 会再次补齐两侧的待定端口 |

约束：

* 同一侧的端口 key 不能重复；
* 新建、加载旧工程、展开自定义算子后，两侧都至少有一个待定端口；
* `bind` 只对**待定**端口生效：端口已有具体类型、端口不存在或 `data_type=None` 时返回
  `False`，不会改类型，也不会追加端口。

### 2.3 `NodeModel` 与扁平 `GraphModel`

`NodeModel` 新增两个字段：

```python
parent_frame: str | None = None   # 所属流程框的节点 id；None = 顶层节点
frame: FrameModel | None = None   # 仅流程框节点使用
```

`is_frame` 只有在 `op_id == FRAME_OP_ID` **且** `frame is not None` 时为真。

图的存储仍然是**扁平**的：框内节点、嵌套框节点与顶层节点全部放在同一个
`GraphModel.nodes` 里，只用 `parent_frame` 串成父子链。这样连线、引擎、撤销、复制粘贴
都不需要递归结构，框架只要在需要时沿链查询：

| 方法 | 说明 |
| --- | --- |
| `node_container(node_id)` | 节点直接所属的框 id |
| `direct_children(frame_id)` | 直接子节点 |
| `frame_subtree(frame_id)` | 全部后代（默认不含框自身），按创建顺序做广度优先遍历 |
| `is_inside(node_id, frame_id)` | 是否位于该框的任意层内部（框自身不算） |
| `set_node_parent(node_id, parent_frame)` | 改所属框；会拒绝把框移进自己的内部 |
| `add_node` | 校验 `parent_frame` 必须指向存在的流程框 |

工程序列化时 `NodeModel.to_dict()` 只在 `parent_frame` 非空时写出该字段，`frame` 存在时
总是写出 `FrameModel.to_dict()`。`GraphModel.from_dict()` 会修复悬空的 `parent_frame`：
父节点不存在或父节点不是流程框时，节点回退为顶层，避免旧工程或手工改坏的 JSON 让整棵树失效。

### 2.4 端口层级与内外侧规则

`GraphModel.port_level(node_id, port_key, role)` 决定一个端口在作为源（`"src"`）或
目标（`"dst"`）时属于哪一层：

| 端口 | `role="src"` 的层级 | `role="dst"` 的层级 |
| --- | --- | --- |
| 普通节点端口 | 节点的 `parent_frame` | 节点的 `parent_frame` |
| 流程框输入端口 | **框 id**（内侧：作为框内数据源） | **框的 `parent_frame`**（外侧：接收框外数据） |
| 流程框输出端口 | **框的 `parent_frame`**（外侧：向框外供数） | **框 id**（内侧：接收框内数据） |

只有**层级相同**的两个端口才允许连线，这就是「框内节点不能直接连到框外节点」的模型层保证。
`connection_allowed()` 返回 `(是否允许, 中文原因)`，失败原因例如：

> 只能连接同一流程框层级内的节点；框内节点请通过流程框的输入/输出端口连接

`connection_role()` 进一步把连线归类为：

| 返回值 | 含义 |
| --- | --- |
| `normal` | 普通节点之间的连线 |
| `frame_input_inner` | 框输入端口（内侧）→ 框内节点 |
| `frame_output_inner` | 框内节点 → 框输出端口（内侧） |
| `frame_input_outer` | 框外节点 → 框输入端口（外侧） |
| `frame_output_outer` | 框输出端口（外侧）→ 框外节点 |
| `invalid` | 节点或端口不存在 / 不是流程框端口 |

### 2.5 `resolve_source` 与跨框解析

`GraphModel.resolve_source(src_node, src_port)` 把可能位于流程框边界上的源端口解析成
**真实普通节点端口**，供引擎建立依赖：

* 源是**流程框输入端口**：沿该输入端口的外侧入边继续解析（框内拿到的数据来自框外上游）；
* 源是**流程框输出端口**：沿该输出端口的内侧入边继续解析（框外拿到的数据来自框内上游）；
* 源是普通节点：直接返回 `(node_id, port_key)`；
* 遇到断开的边界（端口没有连线）或解析成环：返回 `None`。

`frame_output_sources(frame_id)` 对框的每个输出端口取内侧入边并调用 `resolve_source`，
返回 `{输出端口 key: (真实节点 id, 真实端口 key)}`。这个映射就是引擎里框输出伪输入的来源。

### 2.6 回环检测

流程框的边界边（`frame_input_inner` / `frame_output_inner`）不是框节点自己产生的数据依赖，
如果直接算作框节点的出边，「框输入 → 框内节点 → 框输出」会被误判成环。因此：

* `GraphModel.effective_successors()` 在回环检测时忽略 `frame_input_inner` 边；
* `add_connection` 对这两类边界边跳过 `_reachable` 检查；
* 框内节点之间、框外节点之间仍然严格保持 DAG。

### 2.7 端口自动增长如何触发

连线写入模型时，`add_connection` 会调用 `_apply_frame_port_binding(connection)`：

1. 若目标节点是流程框，且目标端口属于框的输入 / 输出端口集合，继续；
2. 用界面层注入的 `_port_type_resolver(src_node, src_port, False)` 查询源端口类型
   （`GraphView` 注入，综合注册表与流程框端口）；
3. 调用 `FrameModel.bind(dst_port, is_input, data_type)`：把待定端口定类型并追加新的待定端口；
4. 端口发生变化时对框节点 `touch()` 并发 `node_changed`，画布重建端口与连线。

因此**只有作为连线目标的框端口会确定类型**。如果源端口本身还是万能 / 待定端口，解析不到
具体类型，框端口保持待定。

撤销方面：`AddConnectionCommand` 在重做前保存 `GraphModel.frame_state_snapshot()`，撤销时
`restore_frame_state()` 把框端口整体恢复（包括被自动追加的待定端口）。普通删除连线不会
回退端口类型或减少端口——只有撤销那次“添加连线”才会。

## 3. 流程框定义（`frame_definition`）

### 3.1 结构

流程框定义是一个可 JSON 序列化的字典（保存成自定义算子时也会写进生成文件）：

```python
{
    "format": "mineimage-frame-definition",
    "version": 1,
    "name": "我的预处理",
    "inputs": [
        {"key": "input_1", "label": "输入 1", "data_type": None},
        {"key": "input_2", "label": "输入 2", "data_type": "image"},
    ],
    "outputs": [
        {"key": "output_1", "label": "输出 1", "data_type": "image"},
    ],
    "nodes": [
        {
            "id": "n1", "op_id": "filter.gaussian_blur", "title": "高斯模糊",
            "pos": [40.0, 80.0], "params": {"ksize": 5}, "parent_frame": None,
        },
        {
            "id": "n2", "op_id": "__frame__", "title": "子流程框",
            "pos": [120.0, 100.0], "params": {}, "parent_frame": None,
            "frame": {"inputs": [], "outputs": []},
        },
    ],
    "connections": [
        {"src_node": "__frame__", "src_port": "input_1", "dst_node": "n1", "dst_port": "image"},
        {"src_node": "n1", "src_port": "image", "dst_node": "__frame__", "dst_port": "output_1"},
    ],
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `format` / `version` | 格式名与版本；更高版本会拒绝加载 |
| `name` | 流程框显示名，也是生成算子的默认名称 |
| `inputs` / `outputs` | 框边界端口列表，元素为 `FramePort.to_dict()` |
| `nodes[].id` | 定义内部的节点 id，只需在本次定义内唯一 |
| `nodes[].op_id` | 普通算子 id，或 `"__frame__"`（嵌套的手工流程框） |
| `nodes[].title` | 节点标题 |
| `nodes[].pos` | **相对流程框左上角**的坐标；展开时叠加落点偏移 |
| `nodes[].params` | 节点参数；保存前会转成 JSON 友好结构（NumPy 标量等会降级） |
| `nodes[].parent_frame` | 所属定义节点 id；`null` 表示框的直接子节点 |
| `nodes[].frame` | 嵌套**手工流程框**的端口定义（`op_id="__frame__"` 时） |
| `nodes[].frame_ref` | 引用另一个自定义流程框算子，见 3.3 |
| `connections` | 连线列表；节点 id 为 `"__frame__"` 时代表流程框自身边界 |

根流程框自身在定义里用哨兵 `"__frame__"` 表示：

* `src_node="__frame__"`：数据从框的输入端口流向框内节点；
* `dst_node="__frame__"`：框内节点的输出流向框的输出端口。

### 3.2 构建与规范化

`build_frame_definition(graph, frame_node_id, name)` 从现有图导出定义：

* 只包含该框的后代节点；如果某段子树属于「引用的自定义算子」（`FrameModel.operator_id` 非空），
  不会再展开它的内部节点；
* 节点位置减去框的左上角坐标，转为相对坐标；
* 框上所有输入 / 输出端口原样写入 `inputs / outputs`；
* 两端都在本次定义范围内的连线才写入 `connections`，框自身映射为 `"__frame__"`；
* 嵌套的手工框写 `frame`；引用的自定义算子只写 `frame_ref`（operator_id / name / 端口），
  避免把被引用算子的实现或内部节点复制进新文件；
* 框内包含**原始图像节点**时抛出 `FrameDefinitionError`，提示先把源图像节点移出框。

`normalize_frame_definition(data)` 在生成文件与展开时做校验和规范化：

* 必须是字典；`format` 为 `None` 或正确格式名；`version` 不能高于当前版本；
* 至少有一个输出端口（输入可以为空）；
* 节点 id、op_id 必须存在且不重复，`"__frame__"` 不能作为普通节点 id；
* 非法节点 / 端口 / 连线会被跳过，而不是让整份定义不可用；
* 端口数据类型无法识别时回退为待定；
* 返回结构固定的新字典，避免直接引用文件里的原始对象。

### 3.3 `frame_ref` 依赖引用

当框内包含一个「从算子库拖入的自定义流程框」时，该子树不会被复制进新算子，而是保存成引用：

```python
{
    "id": "n3",
    "op_id": "frame.my_subflow",
    "title": "子流程",
    "pos": [200.0, 40.0],
    "params": {},
    "parent_frame": None,
    "frame_ref": {
        "operator_id": "frame.my_subflow",
        "name": "子流程",
        "inputs":  [{"key": "input_1", "label": "输入 1", "data_type": "image"}],
        "outputs": [{"key": "output_1", "label": "输出 1", "data_type": "image"}],
    },
}
```

展开时的行为：

1. 注册表里有 `operator_id` 对应的流程框算子 → 递归展开它的定义，生成嵌套流程框；
2. 注册表里没有（依赖缺失 / 文件被删） → 使用 `frame_ref` 里的端口生成一个**占位流程框**，
   `FrameModel.operator_id` 保留原 id，便于画布标记「依赖缺失」并在替换后恢复。

## 4. 保存为自定义算子

### 4.1 `save_frame_operator`

```python
save_frame_operator(graph, frame_node_id, name, *, registry=None, directory=None)
    -> SavedFrameOperator(path, op_id, name, definition)
```

* 默认目录为 `default_user_operator_dir() / "custom"`，即源码运行时
  `operators/custom/`；`OperatorRegistry` 会递归扫描到它；
* 算子名去首尾空白，不能为空，长度上限 64；
* 生成 id 为 `frame.<slug>`，其中 `slug` 由名称中的非字母数字下划线字符转换而来
  （纯中文名称会回退为 `frame`）；若 id 已注册或同名文件存在，依次尝试
  `frame.<slug>_2`、`_3`……；
* 文件名 = `op_id.replace(".", "_") + ".py"`，例如 `frame_my_frame.py`；
* 使用临时文件 + `os.replace` 原子落盘，避免写一半的文件被注册表扫描到；
* 返回生成文件路径、最终 id、名称与定义字典。

### 4.2 生成文件

```python
"""由流程框“我的预处理”保存的自定义算子，内部包含 3 个节点。

本文件由 MineImage「流程框 → 保存为新算子」自动生成。
内部节点以 operator id 的形式引用，不重复其它算子的实现代码；依赖的算子缺失时
本文件仍可正常加载，展开后的流程框内对应节点会显示为失效，可右键替换。
"""

import json

from app.core.data_types import DataType          # 只有端口类型不是待定时才会导入
from app.operators.frame import FrameOperator
from app.operators.spec import OperatorSpec, PortSpec

FRAME_DEFINITION = json.loads('{"format":"mineimage-frame-definition","version":1,...}')


class MyFrame(FrameOperator):
    spec = OperatorSpec(
        id='frame.my_frame',
        name='我的预处理',
        category=('用户自定义算子',),
        description='由流程框“我的预处理”保存的自定义算子，内部包含 3 个节点。',
        version='1.0.0',
        inputs=(
            PortSpec('input_1', '输入 1', any_type=True),
        ),
        outputs=(
            PortSpec('output_1', '输出 1', data_type=DataType.IMAGE),
        ),
    )
    frame_definition = FRAME_DEFINITION


OPERATOR = MyFrame
```

说明（实际生成的文件可能因名称、端口类型不同而略有差异）：

* 生成文件**只导入** `json`、`FrameOperator`、`OperatorSpec / PortSpec`，必要时导入
  `DataType`；**不会 import 任何被引用的算子模块**，也没有 `numpy` / `cv2` 等业务依赖；
* 被引用的自定义流程框只以 `frame_ref` 数据出现，不会内联其实现；
* `FrameOperator.process()` 固定抛出 `OperatorError`：流程框算子只能作为容器从算子库拖入
  画布展开，不能作为普通算子直接执行；
* 端口定义由保存时框的端口快照决定：待定端口生成 `any_type=True`，已定类型生成对应
  `data_type`。

### 4.3 加载、展开与依赖缺失

* `OperatorRegistry` 按普通 `.py` 文件加载生成文件；文件只有数据，不 import 依赖，因此
  **即使内部引用的算子不存在，也能正常注册**；
* `is_frame_operator()` 识别 `FrameOperator` 实例 / 子类，或带有非空
  `frame_definition` 的算子；
* 从算子库拖入画布时，`GraphView.add_operator_at()` 检测到流程框算子，调用
  `materialize_frame_operator()` 展开成「流程框 + 内部节点 + 连线」，整次操作一步撤销；
  落点在已有流程框内时，新框会成为它的嵌套子框；
* 展开的普通节点使用新的节点 id，`parent_frame` 按定义重映射，连线也重新生成；
* 某个普通节点的 `op_id` 不在注册表里 → 该节点 `NodeItem` 显示「失效」徽标；
* 被引用的自定义流程框缺失 → 生成占位框，框标题栏显示「依赖缺失」；
* 右键失效节点 / 占位框 →「替换算子…」打开 `ReplaceOperatorDialog`，列表排除流程框算子；
  替换保留节点 id、位置与所属框，尽量按端口类型重连旧连线，整次一步撤销。若替换的是
  占位流程框，替换前会把它的直接子节点挂到框的父级，避免误删。

## 5. 引擎执行

### 5.1 快照扁平化

`ExecutionEngine._capture_graph()` 抓取快照时并不把流程框当成普通算子，而是：

1. 复制所有节点（含 `parent_frame` 与 `FrameModel.copy()`）；
2. 遍历原始连线：
   * 目标端口是框输入 / 框输出端口的边界边，不直接写入节点 inputs：
     * 框输入边是框外数据流入框的边界，供框内节点解析，框自身不消费；
     * 框输出边由下面的伪输入表达，框自身等待真实来源；
   * 普通连线的源端口用 `GraphModel.resolve_source()` 解析成真实普通节点端口，
     写入 `inputs[(dst_node, dst_port)]`；
3. 对每个流程框，遍历 `frame_output_sources(frame_id)`，把输出端口的真实来源写成
   `inputs[(frame_id, "@out:<port_key>")]`，key 形如 `@out:output_1`；
4. 邻接表 `children` 用于“递归重算下游”的集合扩展。普通结构边保留；
   `frame_input_inner` 边默认不加入（框输入不是框节点的产出），但目标本身是嵌套流程框时保留，
   保证预览外层框会计算内层框。

这样拓扑排序、并行调度、缓存键、等待上游 future 全部使用真实节点 id，流程框只是多了
几条「输出端口 → 真实来源」的伪输入。

### 5.2 `_compute_frame`

`_compute_frame` 只负责组装框的输出，不执行图像处理：

1. 对 `FrameModel.outputs` 逐个查找伪输入 `(框 id, "@out:<port>")`；没有内部来源的输出端口
   直接跳过（新建框允许暂时没有输出）；
2. 等待真实来源节点的 future；上游失败时框失败：
   `上游节点执行失败（端口“xxx”），无法计算当前节点`；
3. 来源结果里没有对应输出端口时框失败：`上游节点未提供输出端口“xxx”的数据`；
4. 按 `FramePort.port_spec()` 调用 `coerce_output_value()` 规范化；失败时报
   `流程框输出“xxx”的数据无效：...`；
5. 组装 `NodeResult(node_id=框 id, outputs={输出端口 key: 值}, port_labels=...)`；
   未命名的图像输出自动命名为 `<框标题>:<端口 key>`。

框内普通节点仍按普通节点独立执行，并通过 `node_finished` 回传结果，所以画布和参数面板可以
单独预览框内节点。框本身不产生额外的输入数据。

### 5.3 缓存、失效与错误隔离

* 缓存 / 失效仍按真实节点 id 进行；框节点自身的 `revision` 来自 `NodeModel.touch()`，
  端口增长会触发节点变化；
* 递归重算下游时使用原始图 `descendants`，可能多失效一些节点（例如框输入上游变化会连带
  框内读取该输入的节点），这是允许的；
* 框内某个节点失败只让该节点标红；框输出等待它时会失败并给出中文原因，其它分支不受影响；
* `.mi` 工程的旧节点没有 `parent_frame` / `frame` 字段时按普通节点加载；`frame` 缺失的
  `__frame__` 节点会在 `NodeModel.from_dict` 中补一个默认 `FrameModel`。

## 6. 画布实现

### 6.1 `FrameItem`

`app/graph/frame_item.py` 中的 `FrameItem` 继承 `NodeItem`：

* 圆角容器 + 标题栏 + 状态徽标；背景 `z=-3`（低于连线），选中时 `z=-2.8`，不会盖住框内连线；
* 输入端口在顶边、输出端口在底边；每个外侧端口有一个同款内侧端口：
  输入内侧端口在 `y=+14px` 处、输出内侧端口在 `y=height-14px` 处；
* 待定端口显示灰色 / 虚线样式，标签追加「（待定）」；框标题栏在任一侧有待定端口时显示
  「待定」徽标；依赖缺失时显示「依赖缺失」并画虚线边框；
* 尺寸由视图根据**直接子节点**的包围盒反推：左右各留 26px，顶部 52px、底部 52px；
  无子节点默认 420×220；有子节点时最小 320×180；
* 框的位置 = 子节点包围盒左上角减去 padding；框本身不单独保存尺寸，尺寸由内容决定；
* `rebuild_ports()` 在端口 key / 类型签名变化时重建端口、几何、连线与 tooltip；
* 框内为空时绘制提示「将算子拖入流程框，组合为可复用的整体」。

### 6.2 `GraphView` 交互

| 操作 | 实现 / 行为 |
| --- | --- |
| 空白处右键「添加流程框」 | `add_frame_at(scene_pos)`；默认标题「流程框」，各一个待定端口 |
| 从算子库拖入框区域 | 落点取**最深命中的嵌套框**作为 `parent_frame`；拖入自定义流程框算子时展开成完整框 |
| 拖到端口附近自动连线 | 新节点的作用域由目标端口层级决定（`_scope_for_port_target`）；跨层时拒绝并提示 |
| 拖动普通节点结束 | 按节点中心命中的最深流程框更新 `parent_frame`；原始图像节点始终留在顶层；进入 / 移出框时**不做**自动吸附连线 |
| 拖动流程框本体 | 实时同步移动全部后代（含嵌套框）；释放时位置变化合并进同一个撤销宏 |
| 删除选中 / 删除框及子节点 | 框 + 全部后代 + 相关连线，一步撤销；按深度从深到浅删除，保证撤销顺序可恢复 |
| 复制 / 粘贴框 | 复制时自动带上全部后代与内部连线；粘贴时重映射 `parent_frame`，父框没被一起复制则回退为顶层 |
| 自动布局 | 只排布 `parent_frame is None` 的顶层节点；框内节点保持相对位置，跟随框整体移动 |
| 框右键「保存为新算子…」 | 命名后调用 `save_frame_operator`；成功后发 `operators_changed` 通知主窗口刷新算子库 |
| 框内节点右键「移出流程框」 | `move_out_of_frame`，保留位置与已有连线，一步撤销 |
| 失效节点 / 占位框右键「替换算子…」 | 打开 `ReplaceOperatorDialog`；替换可撤销 |
| 框端口重建 / 连线刷新 | `_sync_frame_geometries()` 由内向外按深度处理嵌套框，并更新相关连线路径 |

其他细节：

* 框内节点坐标始终是**场景绝对坐标**；移动框时直接平移后代坐标，不需要坐标换算；
* 框的几何同步不写入撤销栈；用户拖动框产生的 `MoveNodeCommand` 才是撤销记录；
* `ports_compatible()` 在原有类型 / 颜色空间校验之后追加
  `GraphModel.connection_allowed()` 的层级校验；
* 失效判定由 `GraphView.is_operator_registered()` 提供：普通节点看 `op_id`，引用的流程框
  看 `FrameModel.operator_id`；
* 画布拖入的流程框算子展开后，节点、连线、框端口自动增长均为同一个撤销宏。

## 7. 算子库、替换对话框与分类

* `CATEGORY_ORDER` 的第一项是 `FRAME_CATEGORY`（「用户自定义算子」），
  `OperatorRegistry.tabs()` 按该顺序排序，所以流程框生成的算子固定显示在算子库最顶部；
* 生成文件里的 `OperatorSpec.category` 也是 `("用户自定义算子",)`；即使文件名 / 目录不同，
  分类仍由声明决定；
* `app/ui/library.py` 的说明浮层对 `FrameOperator` 显示「流程框算子」标识，并提示
  「由流程框保存生成，拖入画布后展开为流程框与内部节点」，不显示空参数表；
* `app/ui/replace_operator.py` 中的 `ReplaceOperatorDialog`：
  * 数据来自 `registry.tabs()`，分类顺序与算子库一致；
  * 默认排除流程框算子，但保留当前被替换的算子（显示「当前」）；
  * 支持跨分类搜索（名称 / id / 描述 / 分类 / 关键字），双击或点「替换」确认；
  * `exec()` 返回是否确认，`selected_op_id()` 返回目标算子 id。

## 8. 文件与测试索引

| 文件 | 职责 |
| --- | --- |
| `app/core/frame.py` | `FramePort` / `FrameModel`、默认端口、绑定与增长、定义构建 / 规范化 |
| `app/core/model.py` | `parent_frame` / `frame`、端口层级、连线层级校验、`resolve_source`、工程序列化 |
| `app/core/constants.py` | `FRAME_OP_ID`、`FRAME_CATEGORY`、`CATEGORY_ORDER` |
| `app/operators/frame.py` | `FrameOperator`、`materialize_frame_operator`、`save_frame_operator`、生成文件渲染 |
| `app/graph/frame_item.py` | `FrameItem` 绘制、内外侧端口、框几何 |
| `app/graph/view.py` | 新建框、拖入 / 嵌套、移动 / 删除 / 复制、保存菜单、替换入口、几何同步 |
| `app/graph/commands.py` | 删除框、改父框、替换算子、连线端口快照恢复等撤销命令 |
| `app/engine/engine.py` | 快照扁平化、`@out:` 伪输入、`_compute_frame` |
| `app/ui/library.py` | 算子库分类顺序、流程框算子说明 |
| `app/ui/replace_operator.py` | 替换算子选择对话框 |
| `tests/test_frame.py` | 模型、端口增长、串行化、定义生成、生成文件可加载、分类 |
| `tests/test_frame_engine.py` | 框执行、嵌套、输出类型、依赖缺失 |
| `tests/test_frame_integration.py` | 画布 → 引擎 → 保存 / 复用端到端 |
| `tests/test_graph.py`、`tests/test_widgets.py` | 画布流程框交互、库分类与替换对话框 |

## 9. 不变量与已知限制

必须保持的不变量：

1. 流程框每侧至少有一个**待定**端口（新建、加载、展开后都成立）；
2. 只有连线**目标**是框端口时才确定类型；待定端口一旦绑定不再自动改回；
3. 普通节点只能与同一层级的端口相连；跨层数据必须经过框的输入 / 输出端口；
4. 框定义至少有一个输出端口；不能包含原始图像节点；
5. 图仍为扁平存储，`parent_frame` 必须指向存在的流程框；加载时修复悬空父框；
6. 流程框算子文件只保存数据、不 import 依赖，确保依赖缺失时仍可注册；缺失依赖在画布上
   以「失效 / 依赖缺失」提示，由用户替换。

已知限制：

* 删除框端口上的连线不会自动回收端口类型或减少端口数量（撤销添加连线才可以）；
* 节点拖入 / 拖出流程框不会自动断开旧连线，旧连线若跨层级会继续保留，但新建连线会被
  层级规则拒绝；需要时手动删除旧连线；
* 自动布局不排布框内节点；
* 流程框算子不能通过 `process()` 直接执行，必须从算子库拖入画布展开；
* 修改被引用的自定义算子后，已经展开到画布的引用框不会自动更新；重新拖入即可使用被引用算子的最新定义。

## 10. 调试

```powershell
$env:PYTHONPATH = "."
python -m unittest tests.test_frame tests.test_frame_engine tests.test_frame_integration -v
python -m unittest tests.test_graph tests.test_widgets -v
```

无界面检查生成文件是否可加载：

```python
from app.operators.loader import OperatorRegistry

registry = OperatorRegistry.build(["operators/custom"])
print(registry.issues)
entry = registry.entry("frame.my_frame")
assert entry is not None
assert entry.operator.frame_definition["format"] == "mineimage-frame-definition"
```

手工调试建议：

* 端口类型不对：先看源端口自身类型（待定万能端口不会触发绑定），再看
  `GraphView._resolve_port_type` / `FrameModel.bind`；
* 框输出为空：确认框输出内侧端口是否接了框内节点，以及
  `GraphModel.frame_output_sources()` 是否返回了真实来源；
* 展开后节点失效：核对 `frame_definition.nodes[].op_id` 与注册表 id 是否完全一致，
  按 F5 刷新后再试；
* 保存失败：看画布状态栏提示；框内有源图像节点时先移出；名称或文件写入失败也会给出原因。
