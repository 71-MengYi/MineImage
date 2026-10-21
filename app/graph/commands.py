"""画布使用的撤销 / 重做命令。

所有用户可见的图修改都通过这些 ``QUndoCommand`` 完成，保证模型与画布条目
始终同步（画布条目只监听模型信号，命令本身不直接操作 ``QGraphicsItem``）。

约定
----
* 命令的 ``redo()`` 在第一次入栈时立即执行一次；
* 需要恢复连线时保留原有 ``ConnectionModel`` 的 id，撤销 / 重做后画面稳定；
* 多个节点一起移动时由调用方用 ``beginMacro`` 组合多条 :class:`MoveNodeCommand`。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QUndoCommand

from app.core.model import ConnectionModel, GraphError, GraphModel, NodeModel, new_id


def _copy_node(node: NodeModel) -> NodeModel:
    return NodeModel.from_dict(node.to_dict())


def _copy_connection(connection: ConnectionModel) -> ConnectionModel:
    return ConnectionModel.from_dict(connection.to_dict())


def _same_pos(left: tuple[float, float], right: tuple[float, float], tol: float = 1e-3) -> bool:
    return abs(float(left[0]) - float(right[0])) <= tol and abs(float(left[1]) - float(right[1])) <= tol


class AddNodeCommand(QUndoCommand):
    """向图中添加一个节点。"""

    def __init__(self, graph: GraphModel, node: NodeModel, text: str = "添加节点") -> None:
        super().__init__(text)
        self._graph = graph
        self._node = node
        self._applied = False

    @property
    def node_id(self) -> str:
        return self._node.id

    def redo(self) -> None:  # noqa: D401 - Qt 接口
        if self._node.id not in self._graph.nodes:
            self._graph.add_node(self._node)
            self._applied = True

    def undo(self) -> None:
        if self._applied and self._node.id in self._graph.nodes:
            self._graph.remove_node(self._node.id)
        self._applied = False


class RemoveNodeCommand(QUndoCommand):
    """删除节点；撤销时连同被自动移除的连线一起恢复。"""

    def __init__(self, graph: GraphModel, node_id: str, text: str = "删除节点") -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        node = graph.nodes.get(node_id)
        self._snapshot = _copy_node(node) if node is not None else None
        self._connections: list[ConnectionModel] = []

    @property
    def node_id(self) -> str:
        return self._node_id

    def redo(self) -> None:
        node = self._graph.nodes.get(self._node_id)
        if node is None:
            return
        try:
            removed = self._graph.remove_node(self._node_id)
        except GraphError:
            return
        self._connections = [_copy_connection(connection) for connection in removed]

    def undo(self) -> None:
        if self._snapshot is None:
            return
        if self._node_id not in self._graph.nodes:
            self._graph.add_node(_copy_node(self._snapshot))
        for connection in self._connections:
            if connection.id in self._graph.connections:
                continue
            if connection.src_node not in self._graph.nodes or connection.dst_node not in self._graph.nodes:
                continue
            try:
                self._graph.add_connection(
                    connection.src_node,
                    connection.src_port,
                    connection.dst_node,
                    connection.dst_port,
                    connection.id,
                )
            except GraphError:
                # 恢复期间图结构已被其它命令改变，保持可继续撤销即可。
                continue


class RemoveNodeTreeCommand(QUndoCommand):
    """删除节点及其全部下游子节点，连同相关连线；整次删除一步撤销。"""

    def __init__(self, graph: GraphModel, node_id: str, text: str = "删除节点及子节点") -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        self._sub_commands: list[RemoveNodeCommand] = []

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_ids(self) -> list[str]:
        """本次删除涉及的节点 id（根节点在前）；首次执行前为空列表。"""

        return [command.node_id for command in self._sub_commands]

    def redo(self) -> None:
        if self._node_id not in self._graph.nodes:
            return
        if not self._sub_commands:
            # 根节点先入栈、子节点后入栈；撤销时按相反顺序恢复，每条连线
            # 的快照由更早入栈的一端持有，因此能完整重建子树内的连线。
            self._sub_commands = [
                RemoveNodeCommand(self._graph, node_id)
                for node_id in self._graph.descendants(self._node_id, include_self=True)
            ]
        for command in self._sub_commands:
            command.redo()

    def undo(self) -> None:
        for command in reversed(self._sub_commands):
            command.undo()


class AddConnectionCommand(QUndoCommand):
    """添加一条连线。

    连线落到流程框边界端口时，模型会绑定端口类型并自动追加待定端口；撤销时
    必须把流程框端口状态一并恢复到连线之前的快照。
    """

    def __init__(
        self,
        graph: GraphModel,
        src_node: str,
        src_port: str,
        dst_node: str,
        dst_port: str,
        connection_id: str | None = None,
        text: str = "添加连线",
    ) -> None:
        super().__init__(text)
        self._graph = graph
        self._src_node = src_node
        self._src_port = src_port
        self._dst_node = dst_node
        self._dst_port = dst_port
        self._connection_id = connection_id or new_id("c")
        self._frame_state: dict[str, dict] | None = None

    @property
    def connection_id(self) -> str:
        return self._connection_id

    def redo(self) -> None:
        if self._connection_id in self._graph.connections:
            return
        self._frame_state = self._graph.frame_state_snapshot()
        self._graph.add_connection(
            self._src_node,
            self._src_port,
            self._dst_node,
            self._dst_port,
            self._connection_id,
        )

    def undo(self) -> None:
        if self._connection_id in self._graph.connections:
            self._graph.remove_connection(self._connection_id)
        if self._frame_state is not None:
            self._graph.restore_frame_state(self._frame_state)
            self._frame_state = None


class RemoveConnectionCommand(QUndoCommand):
    """删除一条连线，撤销时用原 id 恢复。"""

    def __init__(self, graph: GraphModel, connection: ConnectionModel, text: str = "删除连线") -> None:
        super().__init__(text)
        self._graph = graph
        self._snapshot = _copy_connection(connection)

    @property
    def connection_id(self) -> str:
        return self._snapshot.id

    def redo(self) -> None:
        if self._snapshot.id in self._graph.connections:
            self._graph.remove_connection(self._snapshot.id)

    def undo(self) -> None:
        snapshot = self._snapshot
        if snapshot.id in self._graph.connections:
            return
        if snapshot.src_node not in self._graph.nodes or snapshot.dst_node not in self._graph.nodes:
            return
        # 输入端口可能已被其它命令重新占用，此时跳过以避免破坏撤销栈。
        existing = self._graph.input_connection(snapshot.dst_node, snapshot.dst_port)
        if existing is not None and existing.id != snapshot.id:
            return
        try:
            self._graph.add_connection(
                snapshot.src_node,
                snapshot.src_port,
                snapshot.dst_node,
                snapshot.dst_port,
                snapshot.id,
            )
        except GraphError:
            return


class MoveNodeCommand(QUndoCommand):
    """移动节点；连续拖动同一节点时合并为一条记录。"""

    _COMMAND_ID = 0x4D4F

    def __init__(
        self,
        graph: GraphModel,
        node_id: str,
        old_pos: tuple[float, float],
        new_pos: tuple[float, float],
        text: str = "移动节点",
    ) -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        self._old_pos = (float(old_pos[0]), float(old_pos[1]))
        self._new_pos = (float(new_pos[0]), float(new_pos[1]))

    @property
    def node_id(self) -> str:
        return self._node_id

    def id(self) -> int:  # noqa: A003 - Qt 接口
        return self._COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802 - Qt 接口
        if not isinstance(other, MoveNodeCommand):
            return False
        if other._node_id != self._node_id:
            return False
        if not _same_pos(other._old_pos, self._new_pos):
            return False
        self._new_pos = other._new_pos
        return True

    def redo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.set_node_pos(self._node_id, self._new_pos)

    def undo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.set_node_pos(self._node_id, self._old_pos)


class SetParentFrameCommand(QUndoCommand):
    """调整节点所属流程框（拖入 / 移出流程框、替换失效流程框时使用）。"""

    def __init__(
        self,
        graph: GraphModel,
        node_id: str,
        old_parent: str | None,
        new_parent: str | None,
        text: str = "调整所属流程框",
    ) -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        self._old_parent = old_parent
        self._new_parent = new_parent

    @property
    def node_id(self) -> str:
        return self._node_id

    def redo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.set_node_parent(self._node_id, self._new_parent)

    def undo(self) -> None:
        if self._node_id in self._graph.nodes:
            try:
                self._graph.set_node_parent(self._node_id, self._old_parent)
            except GraphError:
                # 旧父框可能已被本次操作删除；保持节点可继续撤销即可。
                return


class SetNodeParamsCommand(QUndoCommand):
    """修改节点参数（供参数面板 / 工作区调用）。"""

    def __init__(
        self,
        graph: GraphModel,
        node_id: str,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
        text: str = "修改参数",
    ) -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        self._old_params = dict(old_params or {})
        self._new_params = dict(new_params or {})

    def redo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.set_node_params(self._node_id, dict(self._new_params))

    def undo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.set_node_params(self._node_id, dict(self._old_params))


class RenameNodeCommand(QUndoCommand):
    """重命名节点（上下文菜单使用）。"""

    def __init__(self, graph: GraphModel, node_id: str, old_title: str, new_title: str, text: str = "重命名节点") -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        self._old_title = old_title
        self._new_title = new_title

    def redo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.rename_node(self._node_id, self._new_title)

    def undo(self) -> None:
        if self._node_id in self._graph.nodes:
            self._graph.rename_node(self._node_id, self._old_title)


class ReplaceOperatorCommand(QUndoCommand):
    """把一个失效 / 占位节点原位替换成普通算子。

    * 节点 id、位置、所属流程框保持不变；
    * 旧连线在替换后按端口类型尝试重连，不能重连的自动丢弃（尽量保留）；
    * 若旧节点是流程框，替换前先把直接子节点挂到框的父级，避免子节点被误删；
    * 旧节点状态、子节点归属与旧连线全部快照，整次替换一步撤销。
    """

    def __init__(
        self,
        graph: GraphModel,
        node_id: str,
        *,
        op_id: str,
        title: str,
        params: dict[str, Any] | None = None,
        reusable_connections: list[tuple[str, str, str, str]] | None = None,
        text: str = "替换算子",
    ) -> None:
        super().__init__(text)
        self._graph = graph
        self._node_id = node_id
        node = graph.nodes.get(node_id)
        self._old_state = node.to_dict() if node is not None else None
        self._new_op_id = op_id
        self._new_title = title
        self._new_params = dict(params or {})
        self._new_connections = list(reusable_connections or [])
        self._new_connection_ids: list[str] = []
        self._reparent: list[tuple[str, str | None, str | None]] = []
        self._old_connections: list[ConnectionModel] = []
        if node is not None:
            new_parent = node.parent_frame
            if node.is_frame:
                for child in graph.direct_children(node_id):
                    self._reparent.append((child.id, child.parent_frame, new_parent))
            self._old_connections = [
                _copy_connection(connection)
                for connection in graph.connections.values()
                if node_id in (connection.src_node, connection.dst_node)
            ]
        self._frame_state: dict[str, dict] | None = None
        self._added_connections: list[str] = []

    @property
    def node_id(self) -> str:
        return self._node_id

    def _apply_node_state(self, state: dict[str, Any]) -> None:
        node = self._graph.nodes.get(self._node_id)
        if node is None:
            return
        snapshot = NodeModel.from_dict(state)
        node.op_id = snapshot.op_id
        node.title = snapshot.title
        node.pos = snapshot.pos
        node.params = dict(snapshot.params)
        node.frame = snapshot.frame
        node.parent_frame = snapshot.parent_frame
        node.touch()
        self._graph.node_changed.emit(self._node_id)

    def redo(self) -> None:
        node = self._graph.nodes.get(self._node_id)
        if node is None:
            return
        self._frame_state = self._graph.frame_state_snapshot()
        for connection in list(self._graph.connections.values()):
            if self._node_id in (connection.src_node, connection.dst_node):
                self._graph.remove_connection(connection.id)
        for child_id, _old_parent, new_parent in self._reparent:
            if child_id in self._graph.nodes:
                try:
                    self._graph.set_node_parent(child_id, new_parent)
                except GraphError:
                    continue
        node.op_id = self._new_op_id
        node.title = self._new_title
        node.params = dict(self._new_params)
        node.frame = None
        node.touch()
        self._graph.node_changed.emit(self._node_id)
        self._graph.structure_changed.emit()

        while len(self._new_connection_ids) < len(self._new_connections):
            self._new_connection_ids.append(new_id("c"))
        self._added_connections = []
        for index, (src_node, src_port, dst_node, dst_port) in enumerate(self._new_connections):
            connection_id = self._new_connection_ids[index]
            if connection_id in self._graph.connections:
                self._added_connections.append(connection_id)
                continue
            if src_node not in self._graph.nodes or dst_node not in self._graph.nodes:
                continue
            try:
                self._graph.add_connection(
                    src_node, src_port, dst_node, dst_port, connection_id
                )
                self._added_connections.append(connection_id)
            except GraphError:
                # 端口类型 / 层级不再兼容：丢弃这条旧连线，保持其余连线可用。
                continue

    def undo(self) -> None:
        for connection_id in list(self._added_connections):
            if connection_id in self._graph.connections:
                self._graph.remove_connection(connection_id)
        self._added_connections = []
        if self._old_state is None or self._node_id not in self._graph.nodes:
            return
        self._apply_node_state(self._old_state)
        # 先恢复流程框端口快照，再恢复子节点归属与旧连线，保证层级校验通过。
        if self._frame_state is not None:
            self._graph.restore_frame_state(self._frame_state)
        for child_id, old_parent, _new_parent in reversed(self._reparent):
            if child_id in self._graph.nodes:
                try:
                    self._graph.set_node_parent(child_id, old_parent)
                except GraphError:
                    continue
        for connection in self._old_connections:
            if connection.id in self._graph.connections:
                continue
            if connection.src_node not in self._graph.nodes or connection.dst_node not in self._graph.nodes:
                continue
            try:
                self._graph.add_connection(
                    connection.src_node,
                    connection.src_port,
                    connection.dst_node,
                    connection.dst_port,
                    connection.id,
                )
            except GraphError:
                continue


class InsertNodeOnEdgeCommand(QUndoCommand):
    """在已有连线上插入一个节点。

    组合四步操作：新增节点、删除旧连线、旧上游 → 新节点、新节点 → 旧下游。
    以宏的形式保证一次撤销 / 重做即可完整还原连线。
    """

    def __init__(
        self,
        graph: GraphModel,
        node: NodeModel,
        connection: ConnectionModel,
        node_input_port: str,
        node_output_port: str,
        text: str = "在连线上插入节点",
    ) -> None:
        super().__init__(text)
        self._graph = graph
        self._node = node
        self._connection = _copy_connection(connection)
        self._node_input_port = node_input_port
        self._node_output_port = node_output_port
        self._sub_commands: list[QUndoCommand] = [
            AddNodeCommand(graph, node, "插入节点"),
            RemoveConnectionCommand(graph, connection, "断开旧连线"),
            AddConnectionCommand(
                graph,
                connection.src_node,
                connection.src_port,
                node.id,
                node_input_port,
                text="连接上游",
            ),
            AddConnectionCommand(
                graph,
                node.id,
                node_output_port,
                connection.dst_node,
                connection.dst_port,
                text="连接下游",
            ),
        ]

    @property
    def node_id(self) -> str:
        return self._node.id

    def redo(self) -> None:
        for command in self._sub_commands:
            command.redo()

    def undo(self) -> None:
        for command in reversed(self._sub_commands):
            command.undo()


__all__ = [
    "AddConnectionCommand",
    "AddNodeCommand",
    "InsertNodeOnEdgeCommand",
    "MoveNodeCommand",
    "RemoveConnectionCommand",
    "RemoveNodeCommand",
    "RemoveNodeTreeCommand",
    "RenameNodeCommand",
    "ReplaceOperatorCommand",
    "SetNodeParamsCommand",
    "SetParentFrameCommand",
]
