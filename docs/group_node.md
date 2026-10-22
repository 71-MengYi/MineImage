# 算子组节点（Group Node）实现说明

算子组是画布上的容器节点：它自身带动态数量的输入 / 输出端口，内部可以放普通算子节点，
也可以继续嵌套算子组；整个算子组可以保存成「用户自定义算子」，之后从算子库拖回画布复用。

本文面向修改 MineImage 的开发者，说明算子组的数据模型、序列化格式、画布交互与引擎执行方式。
用户操作说明见 [user_guide.md](user_guide.md) 第 8 节；从算子作者视角看
`GroupOperator` 的写法见 [operator_guide.md](operator_guide.md) 第 12 节。

## 1. 概念与常量

| 名称 | 位置 | 说明 |
| --- | --- | --- |
| `GROUP_OP_ID` | `app/core/constants.py` | 算子组节点的伪算子 id：`"__group__"` |
| `GROUP_BOUNDARY_ID` | `app/core/group.py` | 定义文件里代表“算子组自身边界”的哨兵 id，同样是 `"__group__"` |
| `CUSTOM_OPERATOR_CATEGORY` | `app/core/constants.py` | 算子组生成的自定义算子所属一级分类：`"用户自定义算子"` |
| `GROUP_DEFINITION_FORMAT` | `app/core/group.py` | `"mineimage-group-definition"` |
| `GROUP_DEFINITION_VERSION` | `app/core/group.py` | 当前定义格式版本：`1` |
| `INVALID_LEVEL` | `app/core/group.py` | `GraphModel.port_level` 对非法端口的返回值：`"__invalid__"` |
| `GROUP_OUTPUT_INPUT_PREFIX` | `app/engine/engine.py` | 引擎快照里算子组输出伪输入的 key 前缀：`"@out:"` |

三个容易混淆的概念：

* **端口本身**：`GroupModel.inputs / outputs` 中的 `GroupPort`，保存 key、显示名与数据类型；
  数据类型为 `None` 时是「待定（未知类型）」。
* **外侧端口 / 内侧端口**：画布上同一个连接点绘制成两个圆形端口。外侧端口负责组外连线，
  内侧端口负责组内连线；两者共享同一个 `GroupPort`，图形相同，内侧端口向内偏移 14px。
* **端口层级**：判断两条线能否相连的“作用域”。普通节点、组输入端口、组输出端口在作为
  源或目标时属于不同的层级，详见 2.4。

## 2. 数据模型

### 2.1 `GroupPort`

```python
@dataclass
class GroupPort:
    key: str                    # "input_1" / "output_1"
    label: str                  # "输入 1" / "输出 1"
    data_type: DataType | None  # None = 待定（任意类型）
```

* `pending`：`data_type is None`；`type_text` 返回「任意类型（待定）」或类型中文名。
* `port_spec()`：待定端口返回 `PortSpec(..., any_type=True)`，已定类型返回普通
  `PortSpec(..., data_type=...)`。因此待定端口可以接收任意已知类型。
* `to_dict()` / `from_dict()`：数据类型写成 `"image"`、`"number"` 等值；无法识别的类型
  在加载时按「待定」处理，而不是报错。
* 端口 key 由 `GroupModel` 生成：输入为 `input_<N>`，输出为 `output_<N>`；显示名分别为
  「输入 N」「输出 N」。

### 2.2 `GroupModel`

```python
@dataclass
class GroupModel:
    inputs: list[GroupPort]
    outputs: list[GroupPort]
    operator_id: str | None   # 来自哪个自定义算子文件；手工创建的算子组为 None
```

关键方法：

| 方法 | 行为 |
| --- | --- |
| `create_default(operator_id=None)` | 新建算子组的初始状态：输入、输出各一个待定端口 |
| `ensure_pending(is_input)` | 该侧没有待定端口时追加一个；已有则直接返回 |
| `ensure_pending_ports()` | 输入、输出两侧各确保一个待定端口 |
| `bind(key, is_input, data_type)` | 把待定端口绑定为具体类型；成功时自动追加新的待定端口并返回 `True` |
| `bind_from_spec(key, is_input, spec)` | 从 `PortSpec` 绑定；万能端口、无类型声明时不绑定 |
| `remove_port(key, is_input)` | 删除指定端口；动态回收、撤销恢复时使用 |
| `relabel_auto_ports()` | 按端口位置重新编号自动生成的「输入 N / 输出 N」；手写标签保持不变 |
| `port_spec(key, is_input)` | 查询端口的 `PortSpec` |
| `spec(title)` | 生成 `id=GROUP_OP_ID`、分类为 `("算子组",)` 的动态 `OperatorSpec`，供节点渲染与连线校验 |
| `copy()` / `to_dict()` / `from_dict()` | 深拷贝与序列化；`from_dict` 会再次补齐两侧的待定端口 |

约束：

* 同一侧的端口 key 不能重复；
* 新建、加载旧工程、展开自定义算子后，两侧都至少有一个待定端口；
* `bind` 只对**待定**端口生效：端口已有具体类型、端口不存在或 `data_type=None` 时返回
  `False`，不会改类型，也不会追加端口；
* 已定类型端口在**内外两侧都没有连线**时会被动态回收，空闲待定端口顶替它的 key；
  待定端口本身不参与回收。

### 2.3 `NodeModel` 与扁平 `GraphModel`

`NodeModel` 新增两个字段：

```python
parent_group: str | None = None   # 所属算子组的节点 id；None = 顶层节点
group: GroupModel | None = None   # 仅算子组节点使用
```

`is_group` 只有在 `op_id == GROUP_OP_ID` **且** `group is not None` 时为真。

图的存储仍然是**扁平**的：组内节点、嵌套算子组节点与顶层节点全部放在同一个
`GraphModel.nodes` 里，只用 `parent_group` 串成父子链。这样连线、引擎、撤销、复制粘贴
都不需要递归结构，框架只要在需要时沿链查询：

| 方法 | 说明 |
| --- | --- |
| `node_container(node_id)` | 节点直接所属的算子组 id |
| `direct_children(group_id)` | 直接子节点 |
| `group_subtree(group_id)` | 全部后代（默认不含算子组自身），按创建顺序做广度优先遍历 |
| `is_inside(node_id, group_id)` | 是否位于该算子组的任意层内部（算子组自身不算） |
| `set_node_parent(node_id, parent_group)` | 改所属算子组；会拒绝把算子组移进自己的内部 |
| `add_node` | 校验 `parent_group` 必须指向存在的算子组 |

工程序列化时 `NodeModel.to_dict()` 只在 `parent_group` 非空时写出该字段，`group` 存在时
总是写出 `GroupModel.to_dict()`。`GraphModel.from_dict()` 会修复悬空的 `parent_group`：
父节点不存在或父节点不是算子组时，节点回退为顶层，避免旧工程或手工改坏的 JSON 让整棵树失效。

### 2.4 端口层级与内外侧规则

`GraphModel.port_level(node_id, port_key, role)` 决定一个端口在作为源（`"src"`）或
目标（`"dst"`）时属于哪一层：

| 端口 | `role="src"` 的层级 | `role="dst"` 的层级 |
| --- | --- | --- |
| 普通节点端口 | 节点的 `parent_group` | 节点的 `parent_group` |
| 算子组输入端口 | **算子组 id**（内侧：作为组内数据源） | **组的 `parent_group`**（外侧：接收组外数据） |
| 算子组输出端口 | **组的 `parent_group`**（外侧：向组外供数） | **算子组 id**（内侧：接收组内数据） |

只有**层级相同**的两个端口才允许连线，这就是「组内节点不能直接连到组外节点」的模型层保证。
`connection_allowed()` 返回 `(是否允许, 中文原因)`，失败原因例如：

> 只能连接同一算子组层级内的节点；组内节点请通过算子组的输入/输出端口连接

`connection_role()` 进一步把连线归类为：

| 返回值 | 含义 |
| --- | --- |
| `normal` | 普通节点之间的连线 |
| `group_input_inner` | 组输入端口（内侧）→ 组内节点 |
| `group_output_inner` | 组内节点 → 组输出端口（内侧） |
| `group_input_outer` | 组外节点 → 组输入端口（外侧） |
| `group_output_outer` | 组输出端口（外侧）→ 组外节点 |
| `invalid` | 节点或端口不存在 / 不是算子组端口 |

### 2.5 `resolve_source` 与跨组解析

`GraphModel.resolve_source(src_node, src_port)` 把可能位于算子组边界上的源端口解析成
**真实普通节点端口**，供引擎建立依赖：

* 源是**算子组输入端口**：沿该输入端口的外侧入边继续解析（组内拿到的数据来自组外上游）；
* 源是**算子组输出端口**：沿该输出端口的内侧入边继续解析（组外拿到的数据来自组内上游）；
* 源是普通节点：直接返回 `(node_id, port_key)`；
* 遇到断开的边界（端口没有连线）或解析成环：返回 `None`。

`group_output_sources(group_id)` 对组的每个输出端口取内侧入边并调用 `resolve_source`，
返回 `{输出端口 key: (真实节点 id, 真实端口 key)}`。这个映射就是引擎里组输出伪输入的来源。

### 2.6 回环检测

算子组的边界边（`group_input_inner` / `group_output_inner`）不是算子组节点自己产生的数据依赖，
如果直接算作算子组节点的出边，「组输入 → 组内节点 → 组输出」会被误判成环。因此：

* `GraphModel.effective_successors()` 在回环检测时忽略 `group_input_inner` 边；
* `add_connection` 对这两类边界边跳过 `_reachable` 检查；
* 组内节点之间、组外节点之间仍然严格保持 DAG。

### 2.7 端口自动增长如何触发

连线写入模型时，`add_connection` 会调用 `_apply_group_port_binding(connection)`：

1. 若目标节点是算子组，且目标端口属于组的输入 / 输出端口集合，继续；
2. 用界面层注入的 `_port_type_resolver(src_node, src_port, False)` 查询源端口类型
   （`GraphView` 注入，综合注册表与算子组端口）；
3. 调用 `GroupModel.bind(dst_port, is_input, data_type)`：把待定端口定类型并追加新的待定端口；
4. 端口发生变化时对算子组节点 `touch()` 并发 `node_changed`，画布重建端口与连线。

因此**只有作为连线目标的算子组端口会确定类型**。如果源端口本身还是万能 / 待定端口，解析不到
具体类型，算子组端口保持待定。

删除连线时，`remove_connection()` 会调用 `_reclaim_group_port()` 做动态回收：

1. 只处理算子组边界上的**已定类型**端口（待定端口是下一条连线的入口，不回收）；
2. 用 `group_port_free()` 检查端口内外两侧是否都已没有连线；任一侧仍有连线就保留端口，
   避免删一条线时误拆另一侧；
3. 完全空闲时删除该端口，并让空闲待定端口顶替它的 key（没有空闲待定端口时补一个），
   保证端口编号不会随连接增删不断增长；
4. `relabel_auto_ports()` 按位置重新编号「输入 N / 输出 N」，手写标签保持不变。

因此下一次连接会重新落在待定端点上，而不是在待定端口后面继续追加新点。

撤销方面：所有会删除连线的命令（`RemoveConnectionCommand`、`RemoveNodeCommand`、
`ReconnectGroupMoveCommand`、`DissolveGroupCommand`、`ReplaceOperatorCommand` 等）都在
重做前保存 `GraphModel.group_state_snapshot()`，撤销时 `restore_group_state()` 把算子组
端口整体恢复，再按原 id 重建连线，保证端口回收可逆。

## 3. 算子组定义（`group_definition`）

### 3.1 结构

算子组定义是一个可 JSON 序列化的字典（保存成自定义算子时也会写进生成文件）：

```python
{
    "format": "mineimage-group-definition",
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
            "pos": [40.0, 80.0], "params": {"ksize": 5}, "parent_group": None,
        },
        {
            "id": "n2", "op_id": "__group__", "title": "子算子组",
            "pos": [120.0, 100.0], "params": {}, "parent_group": None,
            "group": {"inputs": [], "outputs": []},
        },
    ],
    "connections": [
        {"src_node": "__group__", "src_port": "input_1", "dst_node": "n1", "dst_port": "image"},
        {"src_node": "n1", "src_port": "image", "dst_node": "__group__", "dst_port": "output_1"},
    ],
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `format` / `version` | 格式名与版本；更高版本会拒绝加载 |
| `name` | 算子组显示名，也是生成算子的默认名称 |
| `inputs` / `outputs` | 组边界端口列表，元素为 `GroupPort.to_dict()` |
| `nodes[].id` | 定义内部的节点 id，只需在本次定义内唯一 |
| `nodes[].op_id` | 普通算子 id，或 `"__group__"`（嵌套的手工算子组） |
| `nodes[].title` | 节点标题 |
| `nodes[].pos` | **相对算子组左上角**的坐标；展开时叠加落点偏移 |
| `nodes[].params` | 节点参数；保存前会转成 JSON 友好结构（NumPy 标量等会降级） |
| `nodes[].parent_group` | 所属定义节点 id；`null` 表示组的直接子节点 |
| `nodes[].group` | 嵌套**手工算子组**的端口定义（`op_id="__group__"` 时） |
| `nodes[].group_ref` | 引用另一个自定义算子组，见 3.3 |
| `connections` | 连线列表；节点 id 为 `"__group__"` 时代表算子组自身边界 |

根算子组自身在定义里用哨兵 `"__group__"` 表示：

* `src_node="__group__"`：数据从组的输入端口流向组内节点；
* `dst_node="__group__"`：组内节点的输出流向组的输出端口。

### 3.2 构建与规范化

`build_group_definition(graph, group_node_id, name)` 从现有图导出定义：

* 只包含该算子组的后代节点；如果某段子树属于「引用的自定义算子」（`GroupModel.operator_id` 非空），
  不会再展开它的内部节点；
* 节点位置减去算子组的左上角坐标，转为相对坐标；
* 算子组上所有输入 / 输出端口原样写入 `inputs / outputs`；
* 两端都在本次定义范围内的连线才写入 `connections`，算子组自身映射为 `"__group__"`；
* 嵌套的手工算子组写 `group`；引用的自定义算子只写 `group_ref`（operator_id / name / 端口），
  避免把被引用算子的实现或内部节点复制进新文件；
* 组内包含**原始图像节点**时抛出 `GroupDefinitionError`，提示先把源图像节点移出算子组。

`normalize_group_definition(data)` 在生成文件与展开时做校验和规范化：

* 必须是字典；`format` 为 `None` 或正确格式名；`version` 不能高于当前版本；
* 至少有一个输出端口（输入可以为空）；
* 节点 id、op_id 必须存在且不重复，`"__group__"` 不能作为普通节点 id；
* 非法节点 / 端口 / 连线会被跳过，而不是让整份定义不可用；
* 端口数据类型无法识别时回退为待定；
* 返回结构固定的新字典，避免直接引用文件里的原始对象。

### 3.3 `group_ref` 依赖引用

当组内包含一个「从算子库拖入的自定义算子组」时，该子树不会被复制进新算子，而是保存成引用：

```python
{
    "id": "n3",
    "op_id": "group.my_subflow",
    "title": "子流程",
    "pos": [200.0, 40.0],
    "params": {},
    "parent_group": None,
    "group_ref": {
        "operator_id": "group.my_subflow",
        "name": "子流程",
        "inputs":  [{"key": "input_1", "label": "输入 1", "data_type": "image"}],
        "outputs": [{"key": "output_1", "label": "输出 1", "data_type": "image"}],
    },
}
```

展开时的行为：

1. 注册表里有 `operator_id` 对应的算子组 → 递归展开它的定义，生成嵌套算子组；
2. 注册表里没有（依赖缺失 / 文件被删） → 使用 `group_ref` 里的端口生成一个**占位算子组**，
   `GroupModel.operator_id` 保留原 id，便于画布标记「依赖缺失」并在替换后恢复。

## 4. 保存为自定义算子

### 4.1 `save_group_operator`

```python
save_group_operator(graph, group_node_id, name, *, registry=None, directory=None)
    -> SavedGroupOperator(path, op_id, name, definition)
```

* 默认目录为 `default_user_operator_dir() / "custom"`，即源码运行时
  `operators/custom/`；`OperatorRegistry` 会递归扫描到它；
* 算子名去首尾空白，不能为空，长度上限 64；
* 生成 id 为 `group.<slug>`，其中 `slug` 由名称中的非字母数字下划线字符转换而来
  （纯中文名称会回退为 `group`）；若 id 已注册或同名文件存在，依次尝试
  `group.<slug>_2`、`_3`……；
* 文件名 = `op_id.replace(".", "_") + ".py"`，例如 `group_my_group.py`；
* 使用临时文件 + `os.replace` 原子落盘，避免写一半的文件被注册表扫描到；
* 返回生成文件路径、最终 id、名称与定义字典。

### 4.2 生成文件

```python
"""由算子组“我的预处理”保存的自定义算子，内部包含 3 个节点。

本文件由 MineImage「算子组 → 保存为新算子」自动生成。
内部节点以 operator id 的形式引用，不重复其它算子的实现代码；依赖的算子缺失时
本文件仍可正常加载，展开后的算子组内对应节点会显示为失效，可右键替换。
"""

import json

from app.core.data_types import DataType          # 只有端口类型不是待定时才会导入
from app.operators.group import GroupOperator
from app.operators.spec import OperatorSpec, PortSpec

GROUP_DEFINITION = json.loads('{"format":"mineimage-group-definition","version":1,...}')


class MyGroup(GroupOperator):
    spec = OperatorSpec(
        id='group.my_group',
        name='我的预处理',
        category=('用户自定义算子',),
        description='由算子组“我的预处理”保存的自定义算子，内部包含 3 个节点。',
        version='1.0.0',
        inputs=(
            PortSpec('input_1', '输入 1', any_type=True),
        ),
        outputs=(
            PortSpec('output_1', '输出 1', data_type=DataType.IMAGE),
        ),
    )
    group_definition = GROUP_DEFINITION


OPERATOR = MyGroup
```

说明（实际生成的文件可能因名称、端口类型不同而略有差异）：

* 生成文件**只导入** `json`、`GroupOperator`、`OperatorSpec / PortSpec`，必要时导入
  `DataType`；**不会 import 任何被引用的算子模块**，也没有 `numpy` / `cv2` 等业务依赖；
* 被引用的自定义算子组只以 `group_ref` 数据出现，不会内联其实现；
* `GroupOperator.process()` 固定抛出 `OperatorError`：算子组只能作为容器从算子库拖入
  画布展开，不能作为普通算子直接执行；
* 端口定义由保存时算子组的端口快照决定：待定端口生成 `any_type=True`，已定类型生成对应
  `data_type`。

### 4.3 加载、展开与依赖缺失

* `OperatorRegistry` 按普通 `.py` 文件加载生成文件；文件只有数据，不 import 依赖，因此
  **即使内部引用的算子不存在，也能正常注册**；
* `is_group_operator()` 识别 `GroupOperator` 实例 / 子类，或带有非空
  `group_definition` 的算子；
* 从算子库拖入画布时，`GraphView.add_operator_at()` 检测到算子组，调用
  `materialize_group_operator()` 展开成「算子组 + 内部节点 + 连线」，整次操作一步撤销；
  落点在已有算子组内时，新算子组会成为它的嵌套子算子组；
* 展开的普通节点使用新的节点 id，`parent_group` 按定义重映射，连线也重新生成；
* 某个普通节点的 `op_id` 不在注册表里 → 该节点 `NodeItem` 显示「失效」徽标；
* 被引用的自定义算子组缺失 → 生成占位算子组，算子组标题栏显示「依赖缺失」；
* 右键失效节点 / 占位算子组 →「替换算子…」打开 `ReplaceOperatorDialog`，列表排除算子组；
  替换保留节点 id、位置与所属算子组，尽量按端口类型重连旧连线，整次一步撤销。若替换的是
  占位算子组，替换前会把它的直接子节点挂到组的父级，避免误删。

## 5. 引擎执行

### 5.1 快照扁平化

`ExecutionEngine._capture_graph()` 抓取快照时并不把算子组当成普通算子，而是：

1. 复制所有节点（含 `parent_group` 与 `GroupModel.copy()`）；
2. 遍历原始连线：
   * 目标端口是组输入 / 组输出端口的边界边，不直接写入节点 inputs：
     * 组输入边是组外数据流入组的边界，供组内节点解析，算子组自身不消费；
     * 组输出边由下面的伪输入表达，算子组自身等待真实来源；
   * 普通连线的源端口用 `GraphModel.resolve_source()` 解析成真实普通节点端口，
     写入 `inputs[(dst_node, dst_port)]`；
3. 对每个算子组，遍历 `group_output_sources(group_id)`，把输出端口的真实来源写成
   `inputs[(group_id, "@out:<port_key>")]`，key 形如 `@out:output_1`；
4. 邻接表 `children` 用于“递归重算下游”的集合扩展。普通结构边保留；
   `group_input_inner` 边默认不加入（组输入不是算子组节点的产出），但目标本身是嵌套算子组时保留，
   保证预览外层算子组会计算内层算子组。

这样拓扑排序、并行调度、缓存键、等待上游 future 全部使用真实节点 id，算子组只是多了
几条「输出端口 → 真实来源」的伪输入。

### 5.2 `_compute_group`

`_compute_group` 只负责组装算子组的输出，不执行图像处理：

1. 对 `GroupModel.outputs` 逐个查找伪输入 `(算子组 id, "@out:<port>")`；没有内部来源的输出端口
   直接跳过（新建算子组允许暂时没有输出）；
2. 等待真实来源节点的 future；上游失败时算子组失败：
   `上游节点执行失败（端口“xxx”），无法计算当前节点`；
3. 来源结果里没有对应输出端口时算子组失败：`上游节点未提供输出端口“xxx”的数据`；
4. 按 `GroupPort.port_spec()` 调用 `coerce_output_value()` 规范化；失败时报
   `算子组输出“xxx”的数据无效：...`；
5. 组装 `NodeResult(node_id=算子组 id, outputs={输出端口 key: 值}, port_labels=...)`；
   未命名的图像输出自动命名为 `<算子组标题>:<端口 key>`。

组内普通节点仍按普通节点独立执行，并通过 `node_finished` 回传结果，所以画布和参数面板可以
单独预览组内节点。算子组本身不产生额外的输入数据。

### 5.3 缓存、失效与错误隔离

* 缓存 / 失效仍按真实节点 id 进行；算子组节点自身的 `revision` 来自 `NodeModel.touch()`，
  端口增长会触发节点变化；
* 递归重算下游时使用原始图 `descendants`，可能多失效一些节点（例如组输入上游变化会连带
  组内读取该输入的节点），这是允许的；
* 组内某个节点失败只让该节点标红；组输出等待它时会失败并给出中文原因，其它分支不受影响；
* `.mi` 工程的旧节点没有 `parent_group` / `group` 字段时按普通节点加载；`group` 缺失的
  `__group__` 节点会在 `NodeModel.from_dict` 中补一个默认 `GroupModel`。

## 6. 画布实现

### 6.1 `GroupItem`

`app/graph/group_item.py` 中的 `GroupItem` 继承 `NodeItem`：

* 圆角容器 + 标题栏 + 状态徽标；背景 `z=-3`（低于连线），选中时 `z=-2.8`，不会盖住组内连线；
* 输入端口在顶边、输出端口在底边；每个外侧端口有一个同款内侧端口：
  输入内侧端口在 `y=+14px` 处、输出内侧端口在 `y=height-14px` 处；
* 待定端口显示灰色 / 虚线样式，标签追加「（待定）」；
  依赖缺失时标题栏显示「依赖缺失」并画虚线边框；
* 尺寸由视图根据**直接子节点**的包围盒反推：左右各留 26px，顶部 52px、底部 52px；
  无子节点默认 420×220；有子节点时最小 320×180；
* 算子组的位置 = 子节点包围盒左上角减去 padding；算子组本身不单独保存尺寸，尺寸由内容决定；
* `rebuild_ports()` 在端口 key / 类型签名变化时重建端口、几何、连线与 tooltip；
* 组内为空时绘制提示「将算子拖入算子组，组合为可复用的整体」。

### 6.2 `GraphView` 交互

| 操作 | 实现 / 行为 |
| --- | --- |
| 空白处右键「添加算子组」 | `add_group_at(scene_pos)`；默认标题「算子组」，各一个待定端口 |
| 从算子库拖入算子组区域 | 落点取**最深命中的嵌套算子组**作为 `parent_group`；拖入自定义算子组时展开成完整算子组 |
| 拖到端口附近自动连线 | 新节点的作用域由目标端口层级决定（`_scope_for_port_target`）；跨层时拒绝并提示 |
| 拖动普通节点结束 | 按节点中心命中的最深算子组更新 `parent_group`；原始图像节点始终留在顶层；进入 / 移出算子组时经组端口自动重连（`ReconnectGroupMoveCommand`） |
| 自动重连的端口分配 | 优先复用**两侧都没有连线的空闲端口**（待定端口优先，兼容旧版本遗留的已定类型空闲端口），待定端口始终保留在列表末尾；多输入 / 多输出节点按端口声明顺序分配，避免组内连线交叉 |
| 删除算子组边界连线 | `GraphModel.remove_connection()` 回收两侧都已空闲的已定类型端口，空闲待定端口顶替其 key；撤销时恢复端口快照并重建连线 |
| 拖动算子组本体 | 实时同步移动全部后代（含嵌套算子组）；释放时位置变化合并进同一个撤销宏 |
| 删除选中 / 删除算子组及子节点 | 算子组 + 全部后代 + 相关连线，一步撤销；按深度从深到浅删除，保证撤销顺序可恢复 |
| 复制 / 粘贴算子组 | 复制时自动带上全部后代与内部连线；粘贴时重映射 `parent_group`，父算子组没被一起复制则回退为顶层 |
| 自动布局 | 先排布顶层节点，再按深度从外到内排布每个算子组的直接子节点；组内节点与普通节点使用同一套布局规则 |
| 算子组右键「保存为新算子…」 | 命名后调用 `save_group_operator`；成功后发 `operators_changed` 通知主窗口刷新算子库 |
| 算子组右键「解散算子组」 | `dissolve_group` → `DissolveGroupCommand`：内部节点放回组的父级、位置不变，边界连线两两改接为直连，一步撤销 |
| 组内节点右键「移出算子组」 | `move_out_of_group`，经组端口的连线改接为直连，保留位置，一步撤销 |
| 失效节点 / 占位算子组右键「替换算子…」 | 打开 `ReplaceOperatorDialog`；替换可撤销 |
| 算子组端口重建 / 连线刷新 | `_sync_group_geometries()` 由内向外按深度处理嵌套算子组，并更新相关连线路径 |
| 从组输入端口拖到组内节点 | 按下外框输入圆点时，`connection_drag_source()` 把起点映射为同 key 的内侧输出端口，直接建立组输入 → 组内节点连线 |
| 连线未命中端口 | 释放点附近有不兼容输入端口时交给 `connect_ports()` 说明原因；落在节点其它区域时提示“未找到可连接的输入端口”，不再静默取消 |

其他细节：

* 组内节点坐标始终是**场景绝对坐标**；移动算子组时直接平移后代坐标，不需要坐标换算；
* 组的几何同步不写入撤销栈；用户拖动算子组产生的 `MoveNodeCommand` 才是撤销记录；
* `ports_compatible()` 在原有类型 / 颜色空间校验之后追加
  `GraphModel.connection_allowed()` 的层级校验；
* 失效判定由 `GraphView.is_operator_registered()` 提供：普通节点看 `op_id`，引用的算子组
  看 `GroupModel.operator_id`；
* 画布拖入的算子组展开后，节点、连线、算子组端口自动增长均为同一个撤销宏。

## 7. 算子库、替换对话框与分类

* `CATEGORY_ORDER` 的第一项是 `CUSTOM_OPERATOR_CATEGORY`（「用户自定义算子」），
  `OperatorRegistry.tabs()` 按该顺序排序，所以算子组生成的算子固定显示在算子库最顶部；
* 生成文件里的 `OperatorSpec.category` 也是 `("用户自定义算子",)`；即使文件名 / 目录不同，
  分类仍由声明决定；
* `app/ui/library.py` 的说明浮层对 `GroupOperator` 显示「算子组」标识，并提示
  「由算子组保存生成，拖入画布后展开为算子组与内部节点」，不显示空参数表；
* `app/ui/replace_operator.py` 中的 `ReplaceOperatorDialog`：
  * 数据来自 `registry.tabs()`，分类顺序与算子库一致；
  * 默认排除算子组，但保留当前被替换的算子（显示「当前」）；
  * 支持跨分类搜索（名称 / id / 描述 / 分类 / 关键字），双击或点「替换」确认；
  * `exec()` 返回是否确认，`selected_op_id()` 返回目标算子 id。

## 8. 文件与测试索引

| 文件 | 职责 |
| --- | --- |
| `app/core/group.py` | `GroupPort` / `GroupModel`、默认端口、绑定与增长、定义构建 / 规范化 |
| `app/core/model.py` | `parent_group` / `group`、端口层级、连线层级校验、`resolve_source`、工程序列化 |
| `app/core/constants.py` | `GROUP_OP_ID`、`CUSTOM_OPERATOR_CATEGORY`、`CATEGORY_ORDER` |
| `app/operators/group.py` | `GroupOperator`、`materialize_group_operator`、`save_group_operator`、生成文件渲染 |
| `app/graph/group_item.py` | `GroupItem` 绘制、内外侧端口、算子组几何 |
| `app/graph/view.py` | 新建算子组、拖入 / 嵌套、移动 / 删除 / 复制、保存菜单、替换入口、几何同步 |
| `app/graph/commands.py` | 删除算子组、改父算子组并重连（`ReconnectGroupMoveCommand`）、解散算子组（`DissolveGroupCommand`）、替换算子、连线端口快照恢复等撤销命令 |
| `app/engine/engine.py` | 快照扁平化、`@out:` 伪输入、`_compute_group` |
| `app/ui/library.py` | 算子库分类顺序、算子组说明 |
| `app/ui/replace_operator.py` | 替换算子选择对话框 |
| `tests/test_group.py` | 模型、端口增长、串行化、定义生成、生成文件可加载、分类 |
| `tests/test_group_engine.py` | 算子组执行、嵌套、输出类型、依赖缺失 |
| `tests/test_group_integration.py` | 画布 → 引擎 → 保存 / 复用端到端 |
| `tests/test_graph.py`、`tests/test_widgets.py` | 画布算子组交互、库分类与替换对话框 |

## 9. 不变量与已知限制

必须保持的不变量：

1. 算子组每侧至少有一个**待定**端口（新建、加载、展开、回收端口后都成立）；
2. 只有连线**目标**是算子组端口时才确定类型；已定类型端口只有在该端口被完整回收后
   才会重新以待定端口的形式复用；
3. 普通节点只能与同一层级的端口相连；跨层数据必须经过组的输入 / 输出端口；
4. 算子组定义至少有一个输出端口；不能包含原始图像节点；
5. 图仍为扁平存储，`parent_group` 必须指向存在的算子组；加载时修复悬空父算子组；
6. 算子组文件只保存数据、不 import 依赖，确保依赖缺失时仍可注册；缺失依赖在画布上
   以「失效 / 依赖缺失」提示，由用户替换。

已知限制：

* 端口回收以「内外两侧都没有连线」为条件：只删除一侧连线时，另一侧仍在使用中的端口
  会保留，避免误删另一条连接；需要两侧都断开后端口才会消失；
* 节点拖入 / 拖出算子组时会自动经组端口重连；如果目标端口已被占用、类型不兼容或
  层级无法规划，对应的跨层连线会被丢弃（整次拖动仍可撤销）；
* 自动布局会重排组内节点；如果用户希望保留手工摆放，可撤销那次自动布局；
* 算子组不能通过 `process()` 直接执行，必须从算子库拖入画布展开；
* 修改被引用的自定义算子后，已经展开到画布的引用算子组不会自动更新；重新拖入即可使用被引用算子的最新定义。

## 10. 调试

```powershell
$env:PYTHONPATH = "."
python -m unittest tests.test_group tests.test_group_engine tests.test_group_integration -v
python -m unittest tests.test_graph tests.test_widgets -v
```

无界面检查生成文件是否可加载：

```python
from app.operators.loader import OperatorRegistry

registry = OperatorRegistry.build(["operators/custom"])
print(registry.issues)
entry = registry.entry("group.my_group")
assert entry is not None
assert entry.operator.group_definition["format"] == "mineimage-group-definition"
```

手工调试建议：

* 端口类型不对：先看源端口自身类型（待定万能端口不会触发绑定），再看
  `GraphView._resolve_port_type` / `GroupModel.bind`；
* 组输出为空：确认组输出内侧端口是否接了组内节点，以及
  `GraphModel.group_output_sources()` 是否返回了真实来源；
* 展开后节点失效：核对 `group_definition.nodes[].op_id` 与注册表 id 是否完全一致，
  按 F5 刷新后再试；
* 保存失败：看画布状态栏提示；组内有源图像节点时先移出；名称或文件写入失败也会给出原因。
