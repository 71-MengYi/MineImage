# MineImage

MineImage 是一个基于 **PySide6 + OpenCV** 的节点式图像处理实验工作台。把算子从
左侧拖入画布、连线成流程，即可在右侧实时查看任意节点的中间结果，并追溯整条处理
链的每个阶段。

![界面布局](assets/icon.png)

## 功能

* **节点画布**：拖拽算子、上下端口连线、靠近端口自动吸附连线、拖到连线上自动插入
  节点、框选、多分支 DAG、纵向自动布局、撤销/重做。
* **算子库**：一级分类以并排按钮展示，更深层级使用多级折叠分组；支持搜索、拖拽和
  单击查看算子说明浮层。
* **实时预览**：单击节点立即显示该节点输出；双击进入参数面板，拖动参数只重算当前
  节点，点击“保存”后才更新下游。后台线程计算、过期任务自动取消。
* **中间结果追溯**：每个节点缓存自己的最新快照，随时点击查看；工程保存时参数与
  连线一并保存；**文件 → 导出当前节点图像**（Ctrl+E）可导出 PNG/TIFF/NPY。
* **预览交互**：保持比例、Ctrl+滚轮缩放、左键拖动、单视图 / 左右对比 / 滑动对比；
  支持点选、矩形 ROI、画线工具供算子声明式调用。
* **状态栏**：行/列坐标（左上角原点）与原始像素值（R/G/B/A，附 dtype）、缩放比例、
  图像尺寸、当前节点、处理耗时。
* **自动发现算子**：把符合规范的 Python 文件放入 `operators/` 或设置中的额外目录，
  启动时或按 F5 自动扫描；扫描失败不影响其它算子，可在“算子 → 查看加载日志”排错。
* **工程文件**：`.mi` 是 zip 压缩容器，内嵌源图像（npz，保真 uint16/float32/alpha），
  换机器打开不丢图。
* **设置持久化**：`config/MineImage.ini`（便携式，可用 `MINEIMAGE_CONFIG` 覆盖）。
* **打包**：提供 PyInstaller onefile 配置与一键脚本。

## 环境

* Python 3.10+（开发验证：3.13）
* PySide6 >= 6.6（验证：6.10.1）
* NumPy、opencv-python、Pillow

```powershell
python -m pip install -e .
python -m app
```

首次启动后：

1. **文件 → 打开图像** 新建一个工作区标签，并生成“原始图像”节点；
2. 从左侧算子库拖入算子，连接端口；
3. 单击节点查看输出，双击节点调整参数；
4. **文件 → 保存工程** 保存为 `.mi`。

示例图像可运行 `python scripts/make_samples.py` 生成到 `sample_images/`。

## 自定义算子（自动发现）

在 `operators/`（或设置里的额外目录）新建 `.py` 文件，定义模块级 `OPERATOR`：

```python
from app.operators.spec import Operator, OperatorSpec, ParamSpec

class MyOp(Operator):
    spec = OperatorSpec(
        id="example.my_op",
        name="我的算子",
        category=("其他", "示例"),          # 一级/二级/... 分类
        description="算子说明，会显示在算子库浮层中。",
        params=(ParamSpec("gain", "增益", "float", default=1.0, min=0.1, max=4.0, step=0.1),),
    )
    def process(self, image, gain=1.0, **_):
        return (image.array.astype("float32") * gain).clip(0, 255).astype("uint8")

OPERATOR = MyOp
```

完整字段、交互参数（points/roi/line）、联动条件与校验规则见
[docs/operator_guide.md](docs/operator_guide.md)，可复制的模板是
[operators/_template.py](operators/_template.py)。

## 工程结构

```
app/
  core/        图像容器、DAG 模型、.mi 工程格式、图像 IO
  operators/   算子规范、自动扫描、内置算子
  engine/      异步执行引擎（快照 / 取消 / 下游重算）
  graph/       QGraphicsView 节点画布、连线、命令、自动布局
  ui/          主窗口、工作区、算子库、预览、参数面板、设置、主题
operators/     用户算子目录（自动扫描）
docs/          架构、用户手册、算子开发指南
packaging/     PyInstaller spec 与一键打包脚本
tests/         unittest 测试（无需 pytest）
```

## 开发与测试

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -p "test_*.py" -v
```

核心逻辑（模型、算子规范、工程文件、引擎、自动布局、窗口控件）均有单元测试；
完整界面可在 `python -m app` 后手动验证。

## 打包

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1
# 或双击 packaging/build_exe.bat
```

产物为 `dist/MineImage.exe`（onefile）。脚本会把 `docs/`、`operators/` 模板和示例
图像复制到 `dist/`。用户算子目录在 exe 旁：`dist/operators/`。

## 当前范围

* 仅支持 DAG，不支持回环（数据结构保留了扩展空间）。
* 参数撤销为“参数编辑会话”粒度：一次面板编辑在保存/返回时合并为一步撤销，
  撤销步数可在设置中配置。
* 降采样设置只影响预览显示，不改变计算得到的像素与工程数据。

## 许可

MIT。
