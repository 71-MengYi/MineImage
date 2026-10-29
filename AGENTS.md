# AGENTS.md

> 面向 AI 编码代理的 MineImage 工作手册。**先读本文件，再按「Agent 行为映射表」只打开目标文件**；不要用全仓扫描代替定位。
>
> 最后核对：2026-10-30 · 项目版本 0.1.0 · 如与代码/测试冲突，以代码和测试为准，并修正本文件。

## 1. 项目速览

- **是什么**：PySide6 + OpenCV 的节点式图像处理实验工作台。左侧算子库、中间 DAG 画布、右侧实时预览；支持算子组、自定义算子、`.mi` 工程、图像 / CSV 文本矩阵导入。
- **技术栈**：Python ≥3.10（开发验证 3.13）、PySide6 ≥6.6、NumPy、opencv-python ≥4.8、Pillow。Windows/PowerShell 是主要开发环境。
- **OpenCV**：目录算子按 OpenCV 5.0 API 设计，同时兼容 4.x；Haar / HOG 仅在 4.x 可用，5.x 人脸检测走 YuNet ONNX。
- **入口**：`python -m app`（`app/__main__.py` → `app/main.py`）。
- **测试**：`unittest`，离屏 Qt；不要求 pytest。
- **数据流**：`MainWindow → WorkspaceWidget → OperatorLibrary / GraphView ↔ GraphModel / ParamPanel / PreviewWidget / ExecutionEngine → OperatorRegistry`。
- **产品边界**：只支持 DAG，不支持回环；`.mi` 是 zip 容器（`project.json` + 内嵌 `images/*.npz`）。

## 2. 省 token 的阅读规则（先看）

1. **先定位，再读文件**：在本文件的行为表找到最接近的任务；只读「先读」列出的文件/小节。
2. **先 grep，再局部读**：用 `rg -n "类名|方法名|中文文案|错误信息" app tests docs` 定位，再用 `offset/limit` 读命中的 20–80 行；不要整读。
3. **大文件默认不整读**：`app/graph/view.py`（约 3.4k 行）、`tests/test_graph.py`（约 2.8k 行）、`app/ui/params.py`、`app/ui/library.py`、`app/ui/preview.py`、`app/engine/engine.py`、`app/ui/main_window.py` 等；先按方法名/测试名 grep。
4. **文档按标题跳读**：`docs/operator_guide.md`（约 1.2k 行）、`docs/user_guide.md`（约 860 行）、`docs/group_node.md`（约 560 行）、`docs/architecture.md`（约 420 行）；`docs/opencv_catalog.md` 是生成产物，只按算子 id/分类 grep。
5. **生成/运行目录默认不要当作源码阅读、修改或提交**：`dist/**`（打包副本，内容可能过时；验证产物时可运行 exe）、`__pycache__/`、`.test_tmp/`、`.pip_tmp/`、`.onefile_tmp/`、`qa_exe_tmp/`、`widgets-tmp-*/`、`tests/.tmp/`、`tests/tmp*/`、`config/`（运行期 ini/log/db；排错时只读目标日志尾部）、`mineimage.egg-info/`、`sample_images/`（由脚本生成；测试/计时可用，不要手改）、`models/`（下载模型）。生成目录清单以 `.gitignore` 为准。
6. **阅读预算**：一个任务通常只需 1–3 个源文件 + 1 个测试文件 + 1 个文档小节。需要第 4 个源文件才能定位时，先停下来重新查表/搜索，不要继续扩大范围。
7. **不要通读无关模块**：跨模块只读 `docs/architecture.md` 的「主要跨模块接口（冻结）」对应小节；不要为了“熟悉项目”打开整个 `app/` 或 `tests/`。

> **权威顺序**：代码 + 测试（行为事实） > `docs/architecture.md` / `docs/group_node.md` / `docs/operator_guide.md`（设计与不变量） > `README.md` / `docs/user_guide.md`（用户可见行为） > `dist/` 副本（忽略）。

## 3. 命令速查

以下命令均在仓库根目录执行。

```powershell
# 安装（可编辑）
python -m pip install -e .

# 运行
python -m app

# 全量测试（离屏，避免测试触发模型下载）
$env:QT_QPA_PLATFORM = "offscreen"
$env:MINEIMAGE_AUTO_DOWNLOAD_MODELS = "0"
python -m unittest discover -s tests -t tests -p "test_*.py" -v

# 单个测试文件
python -m unittest tests.test_engine -v

# 生成算子目录文档 / 样本 / 演示工程 / 文档截图
python scripts/gen_opencv_catalog_doc.py
python scripts/make_samples.py
python examples/make_demo_project.py
$env:QT_QPA_PLATFORM = "offscreen"; python examples/capture_screenshots.py

# 打包（onefile → dist/MineImage.exe）
powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1
```

- 测试一律 `unittest` 风格；不要写 pytest 专属 fixture/mark。`pyproject.toml` 中的 pytest 配置只是可选依赖，不是测试入口。
- 测试离屏、无网络、无模态弹窗；临时文件写到系统临时目录或 `tests/.tmp/`（已忽略），不要提交。
- 先跑目标测试，交付前尽量跑全量；时间不够时至少跑「改动模块 + 相邻集成测试」，并说明没跑哪些。

## 4. 当前项目规则

### 4.1 架构与正确性（必须保持）

- **模型是唯一事实来源**：`GraphModel` / `NodeModel` 持有图数据，画布 item 只是视图。结构性修改（增删节点/连线、分组、替换、删除）必须走 `app/graph/commands.py` 的 `QUndoCommand`，保持可撤销；参数编辑沿用现有“编辑会话”撤销语义，不要另造一套。
- **端口类型单一来源**：`app/core/data_types.py` 定义 `image / number / edges / contours / components`；连线用 `port_types_compatible()` / `connection_error()`，执行前用 `validate_inputs()`，输出用 `coerce_output_value()` / `output_to_dict()`。不要复制/绕过类型规则。
- **执行引擎语义**：`ExecutionEngine` 每次请求抓不可变图快照，按 DAG 拓扑并行；用 `generation + revision` 丢弃过期结果；错误按分支隔离，旧结果不得写回界面。改缓存/失效/重算必须保持这些语义。
- **算子线程安全**：`process()` 可能被多个后台线程并发调用；不得依赖实例可变状态，不得原地修改输入数组；不要假设单线程。不要在 `process()` 里操作 Qt 控件、弹窗、`app.exec()` 或阻塞等待。
- **工程兼容**：`.mi` 当前格式版本 1；旧工程必须能打开（历史连线不做回溯类型校验）。参数 key / 端口 key 是持久化契约，改名会破坏旧工程。
- **颜色 / dtype**：内部按 OpenCV 习惯 BGR/BGRA，灰度 `(H,W)`；支持 uint8/uint16/int16/float32 等，显示层归一化、状态栏显示原始值。不要假定输入只有 uint8。
- **算子组不变量**（改 group 前必读）：每侧至少一个待定端口；只有连线目标是组端口时才绑定类型；普通节点只能同层连接；组定义至少一个输出且不能含原始图像节点；生成的自定义算子文件只保存数据、不 import 依赖。详见 `docs/group_node.md` 第 9 节。
- **只支持 DAG**；不要引入回环。预览降采样只影响显示，不改变计算/工程数据。

### 4.2 代码 / 文档 / 依赖

- 用户可见文案、错误提示、docstring、文档统一中文；错误信息要说明怎么修。
- 跟随现有风格：4 空格、类型标注、新文件沿用 `from __future__ import annotations`；行宽尽量 ≤100；不做无关重排/全文件格式化。
- 改公开接口（`GraphView` / `ExecutionEngine` / `ParamPanel` / `PreviewWidget` 等）必须同步 `docs/architecture.md`「主要跨模块接口（冻结）」和所有调用点。
- 版本号三处同步：`pyproject.toml`、`app/__init__.py`、`app/core/constants.py`（当前 0.1.0）。
- 新增第三方代码/模型：确认许可证与 MIT 兼容，记录来源；不要编造版本号、星标或下载地址。
- 提交信息沿用现有习惯：`feat: 中文摘要` / `fix: 中文摘要`；一个提交只做一件事；不提交生成物、日志、模型、缓存。

### 4.3 测试 / 文档 / 打包

- 测试优先加到现有覆盖文件（见 §7.2 测试地图），不要为同一行为重复建文件；测试必须可离屏运行。
- 新增算子必须：id 全局唯一、中文名称/说明、默认参数合法且可直接运行、`process()` 不原地修改输入、线程安全；目录算子优先复用 `app/operators/builtin/catalog/_cv.py`。
- 改内置算子：至少跑对应 `tests/test_catalog_*.py`；新增目录算子还要把 id 补进 `tests/test_catalog_all.py` 的 `REQUIRED_IDS`，并跑 `tests/test_catalog_all.py`（覆盖默认参数、7 类数组、输入不可变）。
- 新增一级分类：同步 `app/core/constants.py` 的 `CATEGORY_ORDER`，并确认 `tests/test_catalog_all.py` 的分类断言。
- 改算子注册表/目录：运行 `python scripts/gen_opencv_catalog_doc.py` 更新 `docs/opencv_catalog.md`；**不要手改生成文档**。
- 改用户可见行为：同步 `docs/user_guide.md` / `README.md`；改算子写法：同步 `docs/operator_guide.md`；改算子组：同步 `docs/group_node.md` 的不变量。
- 新增运行时依赖：更新 `pyproject.toml`，并检查 `packaging/MineImage.spec` 的 hiddenimports/datas 是否需要同步；新增资源：同步 spec 和 `build_exe.ps1` 的复制列表。
- 打包配置是源码；`dist/`、`build/`、`.tools/` 是产物，不读不写不提交。保留 `rthook_normalize_resources.py`。
- 自定义算子文件在应用进程内执行，只放可信代码；模型下载走官方 OpenCV Zoo，不要绕过校验或改官方 URL。

### 4.4 团队协作写范围（多 agent 时）

> 单人任务可忽略负责人；多人并行时以 `docs/architecture.md` 末尾「文件写入范围（团队协作）」为准。禁止跨范围修改；需要改别人文件先和 Lead 协调。

| 负责人 | 可写范围 |
| --- | --- |
| Lead | `app/core/**`、`app/operators/spec.py`、`app/operators/loader.py`、`app/operators/group.py`、`app/config.py`、`app/ui/main_window.py`、`app/ui/workspace.py`、`app/main.py`、`pyproject.toml`、`tests/test_group.py`、`tests/test_group_integration.py` |
| engine-ops | `app/engine/**`、`app/operators/builtin/**`、`tests/test_engine.py`、`tests/test_operators.py`、`tests/test_group_engine.py` |
| graph-ui | `app/graph/**`、`tests/test_graph.py` |
| shell-ui | `app/ui/preview.py`、`app/ui/library.py`、`app/ui/replace_operator.py`、`app/ui/params.py`、`app/ui/settings_dialog.py`、`app/ui/theme.py`、`tests/test_widgets.py` |
| qa-docs | `docs/**`、`README.md`、`packaging/**`、`examples/**` |

未列出的测试文件跟随被测源文件；无法判断或需要跨范围时先找 Lead，不要直接改。

## 5. Agent 行为映射表

> 验证列中的 `tests.test_x` 等价于 `python -m unittest tests.test_x -v`。找不到对应行时，先按 §2 用 `rg` 定位，再回到本表；不要直接通读整个目录。

| 我想做 / 判断 | 先读（只读这些） | 通常改这里 | 验证 / 注意 |
| --- | --- | --- | --- |
| 启动、运行、复现界面问题 | `README.md`「环境」「开发与测试」；`app/main.py` | `app/__main__.py`、`app/main.py`、`app/config.py`（仅入口/日志） | `python -m app`；`tests.test_integration` |
| 找主窗口 / 菜单 / 标签页 / 最近文件 | `docs/architecture.md`「总体数据流」；`app/ui/main_window.py`（grep 菜单/动作） | `app/ui/main_window.py`、`app/ui/workspace.py` | `tests.test_integration` |
| 改画布交互：拖动、框选、连线、右键、快捷键、自动滚动 | `docs/architecture.md`「总体数据流」画布条；`app/graph/view.py`（grep 方法名，不整读） | `app/graph/view.py`、`app/graph/scene.py`、`app/graph/node_item.py`、`app/graph/port_item.py`、`app/graph/connection_item.py`、`app/graph/commands.py` | `tests.test_graph`（grep 相关测试名） |
| 改自动布局 | `app/graph/layout.py` 顶部 docstring | `app/graph/layout.py`；`app/graph/view.py` 调用点 | `tests.test_layout` |
| 改连线曲线 / 端口几何 / 吸附 | `app/graph/curve.py`、`app/graph/port_item.py` 相关方法；`docs/architecture.md`「总体数据流」画布条 | `app/graph/curve.py`、`app/graph/port_item.py`、`app/graph/connection_item.py` | `tests.test_port_geometry`、`tests.test_graph` |
| 改撤销 / 重做 / 图编辑命令 | `docs/architecture.md`「总体数据流」画布条；`app/graph/commands.py` 相关命令 | `app/graph/commands.py`；`app/graph/view.py` 调用点 | `tests.test_graph`（undo 相关） |
| 改图模型 / 节点 / 连线 / 序列化 | `docs/architecture.md`「总体数据流」模型条、「端口数据类型系统」；`app/core/model.py` 相关类 | `app/core/model.py`、`app/core/project.py`、`app/core/group.py`（涉及组时） | `tests.test_core`、`tests.test_group`、`tests.test_group_integration`；注意旧 `.mi` 兼容与悬空 `parent_group` 修复 |
| 改 `.mi` 工程 / 图像 IO / 矩阵导入导出 | `docs/user_guide.md` §13；`app/core/project.py`、`app/core/imageio.py`、`app/core/matrixio.py` | `app/core/project.py`、`app/core/imageio.py`、`app/core/matrixio.py`、`app/core/types.py` | `tests.test_core`（roundtrip / uint16 / alpha / `TestMatrixIO`） |
| 改执行引擎 / 并发 / 缓存 / 失效 | `docs/architecture.md`「时间与缓存语义」「ExecutionEngine」；`app/engine/engine.py` 顶部 | `app/engine/engine.py` | `tests.test_engine`；组相关 `tests.test_group_engine`；保持 generation/revision 与线程安全 |
| 改端口类型 / 结构化数据 / 连线校验 | `docs/architecture.md`「端口数据类型系统」；`app/core/data_types.py` | `app/core/data_types.py`、`app/operators/spec.py`（声明/校验部分） | `tests.test_data_types`、`tests.test_catalog_all` |
| 新增 / 修改内置算子 | `docs/operator_guide.md` §1–5、§10–11；同类 `app/operators/builtin/catalog/<module>.py` 的 `OPERATORS` 模式 | 对应 `app/operators/builtin/catalog/*.py`（优先）；单文件旧模式用 `app/operators/builtin/*.py`；新一级分类才改 `app/core/constants.py` | 对应 `tests.test_catalog_*.py`；新 id 补进 `tests/test_catalog_all.py` 的 `REQUIRED_IDS`；跑 `python scripts/gen_opencv_catalog_doc.py`；用 `app/operators/builtin/catalog/_cv.py` 做输入/dtype/输出，默认参数可运行，不原地改输入 |
| 改算子加载 / 发现 / 注册 / 分类 / 加载日志 | `app/operators/loader.py` 顶部约定；`docs/operator_guide.md` §2；`docs/opencv_catalog.md`「扩展方式」 | `app/operators/loader.py`、`app/operators/spec.py`、`app/core/constants.py`（分类顺序） | `tests.test_loader_multi`、`tests.test_engine`（RegistryTests）、`tests.test_catalog_all` |
| 改算子组 / 保存为自定义算子 / 占位替换 | `docs/group_node.md` §2、§5、§9；`app/core/group.py`、`app/operators/group.py` 顶部 | `app/core/group.py`、`app/operators/group.py`、`app/graph/group_item.py`、`app/graph/commands.py`、`app/graph/view.py` 相关段、`app/engine/engine.py` 快照段 | `tests.test_group`、`tests.test_group_engine`、`tests.test_group_integration`、`tests.test_graph`；对照 §9 不变量 |
| 改参数面板 / 自定义控件 / 联动 / 校验 | `docs/operator_guide.md` §5–6；`app/ui/params.py`（grep 方法，不整读） | `app/ui/params.py`、`app/ui/param_widgets.py` | `tests.test_param_widgets`、`tests.test_widgets`；用 `normalize_params`，非法时禁用保存 |
| 改预览 / 缩放 / 平移 / 对比 / ROI 拾取 | `docs/architecture.md`「PreviewWidget」；`app/ui/preview.py` 相关方法 | `app/ui/preview.py`；参数回调看 `app/ui/params.py` | `tests.test_widgets`（PreviewWidgetTests） |
| 改算子库 / 分类顺序 / 搜索 / 拖拽 | `docs/architecture.md`「OperatorLibrary」；`app/ui/library.py` 顶部 | `app/ui/library.py`、`app/core/constants.py`（`CATEGORY_ORDER`） | `tests.test_widgets`、`tests.test_catalog_all` |
| 改设置 / 配置键 / 主题 | `docs/user_guide.md` §14；`app/config.py` 对应属性；`docs/architecture.md`「SettingsDialog」「Theme」 | `app/config.py`、`app/ui/settings_dialog.py`、`app/ui/theme.py` | `tests.test_settings_models`、`tests.test_widgets`（主题/设置） |
| 改日志 / 日志查看器 | `docs/architecture.md`「应用日志」；`app/core/logstore.py`、`app/ui/log_dialog.py` 顶部 | `app/core/logstore.py`、`app/ui/log_dialog.py`、`app/main.py`（handler 安装） | `tests.test_logstore`、`tests.test_log_dialog`、`tests.test_log_integration` |
| 改 ONNX 模型下载 / 人脸检测 / DNN 推理 | `README.md`「OpenCV ONNX 模型与自动下载」；`app/core/models.py` 顶部；`docs/user_guide.md` §10.5 | `app/core/models.py`、`app/ui/model_download.py`、`app/operators/builtin/catalog/detection.py`、`app/operators/builtin/catalog/dnn.py` | `tests.test_models`、`tests.test_face_detection`、`tests.test_dnn_infer`；URL 必须来自官方，缺模型时中文提示/回退 |
| 改替换算子 / 失效节点 / 依赖缺失 | `docs/architecture.md`「ReplaceOperatorDialog」；`app/ui/replace_operator.py`；`docs/group_node.md` §4.3 | `app/ui/replace_operator.py`、`app/graph/view.py`（replace 相关） | `tests.test_widgets`（ReplaceOperatorDialogTests）、`tests.test_group_integration` |
| 改打包 / 发布 | `packaging/README.md`；`packaging/MineImage.spec`；`README.md`「打包」 | `packaging/*.spec`、`packaging/build_exe.ps1`、`packaging/launcher.py`、`packaging/rthook_normalize_resources.py` | `packaging/build_exe.ps1` + `dist/MineImage.exe` 启动冒烟；无桌面环境则说明未验证；不要改 `dist/` |
| 更新用户文档 / 文档截图 | `docs/user_guide.md` 对应节；`README.md`；`examples/README.md` | `docs/*.md`、`docs/images/*.png`、`README.md`、`examples/capture_screenshots.py` | 链接可达；截图用离屏脚本；不手改 `docs/opencv_catalog.md` |
| 查 UI 文案 / 错误提示 / 符号在哪 | 直接 `rg -n "文案" app tests docs` | 命中的文件 | 相关测试；对照 `docs/user_guide.md` 故障表 |
| 查设计原因 / 历史 | `docs/architecture.md` 对应节 → `git log --oneline -- <path>`（≤10 条） | 通常不改；要改先确认不变量 | 相关测试 |
| 查算子清单 / 覆盖状态 | `docs/opencv_catalog.md` 的目录/分类（grep id/名称） | 不直接改；改注册表后跑生成脚本 | `python scripts/gen_opencv_catalog_doc.py`、`tests.test_catalog_all` |
| 排查性能 / 大图卡顿 | `docs/architecture.md`「时间与缓存语义」；`docs/operator_guide.md` §10.2 第 6 条；`app/engine/engine.py` 相关 | 算子实现（先缩图/少复制）、引擎 worker/cache、UI 线程减负 | 用 `sample_images` 或临时大图计时；`tests.test_engine` |
| 只想了解项目，不修改 | `README.md` + `docs/architecture.md` | 不改 | 不要读源码全量 |

## 6. 文档地图

| 文档 | 什么时候读 | 怎么读 / 关键节 |
| --- | --- | --- |
| `README.md` | 安装、启动、功能总览、工程结构、打包、当前范围 | 先读「环境」「工程结构」「开发与测试」「当前范围」；不要把它当详细设计 |
| `docs/architecture.md` | 模块边界、数据流、冻结接口、时间/缓存、团队写范围 | 按标题读；改跨模块接口必读「主要跨模块接口（冻结）」和「文件写入范围（团队协作）」 |
| `docs/operator_guide.md` | 写/改算子 | §3–6（Spec/Port/Param/联动）、§10（测试调试）、§11（最佳实践）；§12 仅算子组 |
| `docs/group_node.md` | 算子组 / 自定义算子 | §2 数据模型、§5 引擎执行、§9 不变量、§10 调试；实现细节按需 |
| `docs/user_guide.md` | 用户可见行为、设置、`.mi`、快捷键、故障 | 按节读；改 UI 行为时对照 §8/§13/§14/§17/§18 |
| `docs/opencv_catalog.md` | 算子 id / 分类 / 覆盖状态 | 生成文档，只 grep，不手改；改完注册表跑 `scripts/gen_opencv_catalog_doc.py` |
| `docs/third_party.md` | 节点编辑器选型背景 | 只在重新评估画布技术选型时读；日常任务忽略 |
| `docs/images/*.png` | 文档截图 | 由 `examples/capture_screenshots.py` 生成；不要手改像素，改脚本后重生成 |
| `operators/README.md`、`operators/_template.py` | 用户算子目录约定 / 模板 | 新增用户算子或改模板时读 |
| `packaging/README.md` | 打包细节、运行期布局 | 改打包/发布时读 |
| `examples/README.md` | 示例、演示工程、截图脚本 | 改示例或截图时读 |

## 7. 源码与测试地图

### 7.1 源码

| 路径 | 职责 / 关键入口 |
| --- | --- |
| `app/main.py`、`app/__main__.py` | 启动、日志/异常钩子；`main()` |
| `app/config.py` | `AppSettings`、配置路径、`MINEIMAGE_*` 环境变量 |
| `app/core/` | `model.py` 图/工作区、`data_types.py` 端口类型、`types.py` `ImageData`、`group.py` 算子组、`project.py` `.mi`、`imageio.py` 图像 IO、`matrixio.py` CSV/文本矩阵、`models.py` ONNX、`logstore.py`、`constants.py` |
| `app/operators/` | `spec.py` 算子/参数/端口声明、`loader.py` 扫描注册、`group.py` 算子组保存/展开、`builtin/catalog/*.py` 目录算子、`builtin/*.py` 单文件算子、共享工具 `builtin/_utils.py` / `catalog/_cv.py` |
| `app/engine/engine.py` | `ExecutionEngine`：快照 / 拓扑并行 / 缓存 / generation |
| `app/graph/` | `view.py` 画布交互、`scene.py`、`commands.py` 撤销命令、`layout.py` 自动布局、`group_item.py`、`node_item.py`、`port_item.py`、`connection_item.py`、`curve.py`、`collapse_item.py` |
| `app/ui/` | `main_window.py`、`workspace.py`、`library.py`、`params.py`、`param_widgets.py`、`preview.py`、`settings_dialog.py`、`replace_operator.py`、`log_dialog.py`、`theme.py`、`model_download.py` |
| `operators/` | 用户算子目录（运行时扫描）；`_template.py`；`custom/` 由“保存为新算子”生成 |
| `tests/` | unittest；见 §7.2 |
| `docs/` | 设计 / 用户 / 算子文档；见 §6 |
| `scripts/` | `make_samples.py`、`gen_opencv_catalog_doc.py` |
| `packaging/` | `MineImage.spec`、`build_exe.ps1`、`launcher.py`、`rthook_normalize_resources.py` |
| `examples/` | `demo.mi`、`capture_screenshots.py`、`make_demo_project.py`、自定义算子示例 |
| `assets/`、`sample_images/` | 图标；生成样本（`sample_images/` 不入库） |
| `dist/`、`config/`、`build/`、`.tools/` | 运行/构建产物；不读/不改/不提交 |

### 7.2 测试

| 领域 | 测试文件 |
| --- | --- |
| 核心模型 / 工程 / 矩阵 IO / 算子规范 | `test_core.py` |
| 端口类型 / 结构化数据 / 连线校验 | `test_data_types.py` |
| 目录算子分组 | `test_catalog_filter_geom.py`、`test_catalog_segment_feature.py`、`test_catalog_arithmetic_color.py`、`test_catalog_enhance_draw.py`、`test_catalog_video_analysis.py` |
| 目录算子全量门禁 | `test_catalog_all.py` |
| 内置算子 / 多算子模块 / 输出节点 | `test_operators.py`、`test_loader_multi.py`、`test_output_view.py` |
| 引擎 / 注册表 | `test_engine.py` |
| 画布 / 布局 / 端口几何 | `test_graph.py`、`test_layout.py`、`test_port_geometry.py` |
| 算子组 | `test_group.py`、`test_group_engine.py`、`test_group_integration.py` |
| 主窗口 / 控件 / 参数控件 / 设置 | `test_integration.py`、`test_widgets.py`、`test_param_widgets.py`、`test_settings_models.py` |
| 日志 | `test_logstore.py`、`test_log_dialog.py`、`test_log_integration.py` |
| 模型 / DNN / 人脸 | `test_models.py`、`test_dnn_infer.py`、`test_face_detection.py` |

## 8. 交付前检查

- [ ] `git status` / `git diff` 只含预期文件；没有 `dist/`、缓存、日志、模型、临时文件。
- [ ] 目标测试通过；交付前尽量跑全量，并记录结果。
- [ ] 行为变化已同步文档：用户手册/README；算子文档/生成文档；架构/分组不变量。
- [ ] 没有破坏冻结接口；若改了，`docs/architecture.md` 与所有调用点已同步。
- [ ] 线程安全、undo、`.mi` 兼容性、旧工程参数/端口 key 兼容性已自查。
- [ ] 打包改动已说明是否实际构建/启动验证。
- [ ] 提交信息符合 `feat:` / `fix:` + 中文摘要。

## 9. 维护本文件

- 新增/重命名模块、测试、脚本、文档入口时，同步更新 §1、§5、§6、§7。
- 规则不要复制整篇文档；保持本文件 ≤ ~300 行，细节用链接指向 `docs/`。
- 团队模式下由 Lead 维护本文件；其他成员先提修改建议，不要直接改。
- 若本文件与代码/测试/`docs/architecture.md` 冲突，以代码和测试为准，并修正本文件。
