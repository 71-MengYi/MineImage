# examples 目录

本目录放可运行的示例、演示工程和文档截图脚本。所有示例都**不依赖外部素材**，
在仓库根目录执行即可。

---

## 1. `demo.mi`：演示工程

双击 `demo.mi` 或 **文件 → 打开工程…** 打开，内容为：

```
原始图像 ──> 灰度化 ──> 高斯模糊
```

* 3 个节点、2 条连线、1 张内嵌源图像；
* 演示“原始图 → 灰度 → 模糊”的常见预处理链；
* 源图像以 npz 内嵌在 `.mi` 里，换机器打开不丢图，不依赖 `sample_images/`；
* 打开后自动选中末端节点；单击任意节点即可在右侧看到该节点的中间结果，
  双击可改参数；
* 位置由真实画布的 `auto_layout()` 生成，默认 1500×920、三栏 1:2:4 下三个
  节点都完整可见。

> 多输入 / 多输出端口的完整示例见 `custom_operators/highlight.py`
> （双输入 + 双输出），以及内置的“通道拆分 / 通道合并”算子。

重新生成（会覆盖现有 `demo.mi`）：

```powershell
python scripts/make_samples.py     # 首次需要，生成 sample_images/sample_color.png
python examples/make_demo_project.py
```

脚本最后会用 `load_project()` 回读并校验节点/连线/内嵌图像数量。

> 维护提示：仓库 `.gitignore` 已为 `examples/*.mi` 添加例外，`demo.mi` 可以随
> 源码一起分发。

---

## 2. `custom_operators/highlight.py`：自定义算子示例

一个可直接复制到 `operators/` 使用的完整算子“示例：交互标注”：

* 两个输入：`image`（必填，L/BGR/BGRA）+ `mask`（可选，L）；
* 两个输出：标注图（BGR）+ 高亮蒙版（L）；
* 参数覆盖 `enum / bool / int / float / color / points / roi / line`；
* 演示 `visible_when` 的两种写法（裸变量名与 `params['key']`）；
* 矩形/线段/点集可在右侧预览中“拾取”。

试用：

1. 把 `highlight.py` 复制到 `operators/`；
2. 回到 MineImage 按 **F5**；
3. 在左侧「其他 → 示例」下找到“示例：交互标注”；
4. 连上原图，双击节点，在参数面板中切换矩形/线段/点集并点“拾取”。

字段与写法说明见 [docs/operator_guide.md](../docs/operator_guide.md)。

---

## 3. `capture_screenshots.py`：离屏生成文档截图

用于生成 `docs/images/*.png`（主工作区、参数面板、左右对比、滑动对比、算子说明
浮层）。脚本使用 Qt 的 `offscreen` 平台，不需要真实显示器，也不会调用
`app.exec()` 或弹模态对话框。

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python examples/capture_screenshots.py
```

要点：

* 自动加载 `C:/Windows/Fonts/msyh.ttc`（微软雅黑），避免中文变方框；
* 用 `sample_images/sample_color.png` 作为源图，构造
  “原始图像 → 灰度化 → 高斯模糊”；
* 输出固定 1500×920，PNG 均小于 600 KiB；
* 脚本还会调用 `library.show_operator_info("basic.grayscale")` 生成说明浮层截图；
* 结束前把工作区 `mark_clean()` 并 `shutdown()`，避免关闭时的保存提示。

如果 `sample_images/` 为空，脚本会调用 `scripts/make_samples.py` 里的
`make_color_sample()` 自动生成。

---

## 4. `make_demo_project.py`：工程生成与回读校验

```powershell
python examples/make_demo_project.py
```

流程：

1. offscreen 启动 `MainWindow`，打开 `sample_images/sample_color.png`；
2. 通过 `GraphView.add_operator_at()` 添加灰度化、高斯模糊并连线；
3. 调用 `GraphView.auto_layout()` / `fit_graph()` 按当前界面计算位置；
4. `WorkspaceWidget.save_to()` 写出 `examples/demo.mi`（源图像内嵌）；
5. 重新打开工程，检查节点/连线/内嵌图像数量，并断言三个节点在
   1500×920 下完整可见；
6. 在 `.test_tmp/demo_open.png` 留存验证截图（已 gitignore），并打印节点位置。

---

## 5. 算子组与自定义算子（说明）

本目录**没有**单独的算子组示例文件：算子组需要在画布上创建，保存生成的算子文件属于用户数据，
不适合作为固定示例随仓库分发。可以按下面的步骤在几分钟内试用：

1. 打开 `demo.mi` 或任意已有工程；
2. 在画布空白处右键 → **添加算子组**；
3. 从算子库拖入「灰度化」「高斯模糊」等算子到组内，用组的输入 / 输出内侧端口连好线；
4. 右键算子组 → **保存为新算子…**，例如命名为 `DemoGroup`；
5. 生成文件位于 `operators/custom/group_demogroup.py`，算子 id 为
   `group.demogroup`；保存成功后算子库会自动刷新（也可以按 **F5**）；
6. 在算子库顶部 **用户自定义算子** 分类中找到它，拖回画布即可展开复用。

说明：

* 生成文件只保存内部节点的 operator id、参数与连线，不复制其它算子实现；
* 依赖缺失时文件仍可加载，拖入后缺失节点显示「失效」，可右键「替换算子…」；
* 组内不能保存原始图像节点；已连线节点拖入 / 拖出算子组会自动改接到组端口上。

相关文档：[算子组实现说明](../docs/group_node.md)、
[用户手册第 8 节](../docs/user_guide.md#8-算子组与自定义算子)、
[算子编写指南第 12 节](../docs/operator_guide.md#12-算子组算子groupoperator)。

---

## 6. 相关文档

* [用户手册](../docs/user_guide.md)
* [算子编写指南](../docs/operator_guide.md)
* [架构与模块边界](../docs/architecture.md)
* [第三方节点编辑器对比](../docs/third_party.md)
