# MineImage

MineImage 是一个基于 **PySide6 + OpenCV** 的节点式图像处理实验工作台。把算子从
左侧拖入画布、连线成流程，即可在右侧实时查看任意节点的中间结果，并追溯整条处理
链的每个阶段。

![界面布局](assets/icon.png)

## 功能

* **节点画布**：拖拽算子、上下端口连线、靠近端口自动吸附连线、拖到连线上自动插入
  节点、Ctrl+C/V 复制粘贴节点、Ctrl+拖动框选、拖动到画布边缘自动滚动、多分支节点折叠 /
  展开、多分支 DAG、纵向自动布局、撤销/重做。
* **算子库**：内置 **195 个算子 / 14 个一级分类**，覆盖 OpenCV 5.0 的 core、imgproc、
  photo、features2d、objdetect、video、calib3d、dnn 等主要模块；一级分类为整行折叠
  分组，默认全部收起，可同时展开多个；更深层级使用多级折叠分组；支持搜索、拖拽和
  单击查看算子说明浮层。清单见 [docs/opencv_catalog.md](docs/opencv_catalog.md)。
* **实时预览**：单击节点立即显示该节点输出（无有效快照时后台重算）；双击进入参数
  面板，拖动参数会实时重算当前节点并递归更新全部下游；点击“保存”会强制让该节点与
  下游整棵子树失效重算。一次请求内部按 DAG 拓扑序提交，互不依赖的分支并行跑；
  后台线程数默认 min(4, CPU 核数)，可在设置中调整（1–16）。过期任务自动取消，
  旧结果由 generation 丢弃。
* **中间结果追溯**：每个节点缓存自己的最新快照，随时点击查看；工程保存时参数与
  连线一并保存；**文件 → 导出当前节点图像**（Ctrl+E）可导出 PNG/TIFF/NPY。
* **类型化端口**：节点间可传递图像、数值、边缘特征、轮廓特征、连通域五种数据；
  多输出节点在底部按声明顺序排列端口，端口按类型着色；新建连线时校验数据类型
  （图像端口还校验颜色空间），不匹配的连接会被拒绝，执行前还会再校验一次输入。
  「其他 → 可视化 → 输出预览」使用单个万能入口 / 出口透传任意类型；节点卡片直接显示单个数值或图像缩略图，边缘 / 轮廓 / 连通域等数组详情统一在参数面板的「输出数据」区查看。
* **算子组与自定义算子**：画布右键新建算子组，把多个算子包成一个带动态输入 / 输出端口的
  整体；组内可嵌套、整体移动 / 复制 / 删除，组的尺寸随内部节点自动调整。已连线节点
  拖入 / 拖出算子组时会自动经组端口重连；右键「解散算子组」可保留内部节点与连接关系。
  右键「保存为新算子…」可生成到 `operators/custom/`，归入算子库顶部的「用户自定义算子」
  分类；生成文件只引用内部算子，不复制实现，依赖缺失时仍可加载并替换失效节点。
  详见 [docs/group_node.md](docs/group_node.md)。
* **预览交互**：保持比例、Ctrl+滚轮缩放、左键拖动、单视图 / 左右对比 / 滑动对比；
  支持点选、矩形 ROI、画线工具供算子声明式调用。
* **状态栏**：行/列坐标（左上角原点）与原始像素值（R/G/B/A，附 dtype）、缩放比例、
  图像尺寸、当前节点、处理耗时。
* **自动发现算子**：把符合规范的 Python 文件放入 `operators/` 或设置中的额外目录，
  启动时或按 F5 自动扫描；扫描失败不影响其它算子，可在“算子 → 查看加载日志”排错。
* **OpenCV ONNX 模型**：OpenCV 5 移除了 Haar / HOG 等传统 objdetect API，人脸检测
  改用 YuNet ONNX 模型，DNN 推理预置 MobileNetV2 分类模型；参数面板直接显示官方
  下载地址并支持后台下载，算子也可在运行时自动下载。模型缓存到
  `<应用目录>/models`。
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

端口默认是图像；在 `PortSpec` 里给 `data_type` 写 `"number"`、`"edges"`、
`"contours"` 或 `"components"` 即可声明数值或结构化输出。`process()` 可以直接
返回 `NumberValue`、`EdgesData`、`ContoursData`、`ComponentsData`，也可以返回
裸 `int/float`、`(N,2)` 点集或二维 int32 标签图，框架会按端口类型自动规范化。
多输出用 `{输出端口key: 值}` 字典返回，节点底部会按声明顺序生成多个输出端口。

`process()` 会被后台多个线程并发调用：同一个算子实例可能同时服务多个节点或兄弟
分支，输入数组也可能被多个下游同时读取。因此算子必须线程安全——不要依赖算子
实例上的可变状态（例如用 `self.xxx` 做缓存），不要原地修改输入数组，需要变换时
返回新数组。内置算子均为无状态实现，满足该约束。

完整字段、交互参数（points/roi/line）、联动条件与校验规则见
[docs/operator_guide.md](docs/operator_guide.md)，可复制的模板是
[operators/_template.py](operators/_template.py)。

内置目录算子位于 `app/operators/builtin/catalog/`，按 OpenCV 模块拆分文件；一个
模块可以定义多个算子并用 `OPERATORS = (A, B, ...)` 批量导出（仍兼容单个
`OPERATOR`）。目录算子统一使用 `catalog/_cv.py` 做输入校验、dtype 归一化与输出恢复。

参数面板支持算子作者注册自定义控件与布局（`ParamSpec.widget` /
`ParamSpec.layout` + `app.ui.param_widgets`），详见
[docs/operator_guide.md](docs/operator_guide.md#54-自定义控件与布局)。

算子组也可以直接保存为自定义算子：画布右键算子组 →「保存为新算子…」，生成文件会放到
`operators/custom/`，并自动出现在算子库最顶部的「用户自定义算子」分类。生成文件只保存
内部节点的 operator id / 参数 / 连线，不复制依赖实现；缺依赖时仍可加载，展开后可右键替换
失效节点。文件格式见 [docs/operator_guide.md](docs/operator_guide.md) 第 12 节，
模型与画布实现见 [docs/group_node.md](docs/group_node.md)。

## OpenCV ONNX 模型与自动下载

OpenCV 5.0 移除了 `cv2.CascadeClassifier` 与 `cv2.HOGDescriptor`，因此
“人脸检测”算子改用 OpenCV Zoo 的 **YuNet ONNX** 模型；“DNN 推理”算子在模型
路径留空时预置 **MobileNetV2** 图像分类模型。模型文件不随 `opencv-python`
安装，需要下载后放到本机：

| 模型 | 用途 | 官方下载地址 |
| --- | --- | --- |
| YuNet（约 230 KB） | 人脸检测 | <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx> |
| SFace（约 37 MB） | 人脸特征 / 比对 | <https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx> |
| MobileNetV2（约 13 MB） | DNN 推理 / ImageNet 分类 | <https://github.com/opencv/opencv_zoo/raw/main/models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx> |

模型来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，版权与许可归各模型
作者，使用前请阅读对应模型目录下的 LICENSE。

**下载方式（任选其一）**

1. 双击“人脸检测”或“DNN 推理”节点，在「模型文件」参数下点击 **下载模型**：
   面板显示官方地址、下载进度与本地缓存状态；
2. 保持「自动下载模型 / 自动下载预置模型」开启（默认），首次运行算子时自动从
   OpenCV Zoo 下载；
3. **设置 → 模型** 页集中查看所有内置模型的状态、下载地址并下载；
4. 手动下载后放进 `<应用目录>/models/`，或在参数面板点「浏览…」选择文件。

**缓存位置与环境变量**

* 默认目录：`<应用目录>/models`（源码运行为仓库根目录，打包版为 exe 所在目录）；
* `MINEIMAGE_MODEL_DIR`：覆盖模型缓存目录；
* `MINEIMAGE_AUTO_DOWNLOAD_MODELS=0`：全局关闭自动下载（仍可在参数面板手动点
  「下载模型」）。内网 / 离线环境建议提前把模型文件放到缓存目录。

> OpenCV 4.x 仍自带 Haar 级联：人脸检测在缺少 YuNet 模型时会自动回退 Haar；
> OpenCV 5.x 不再提供 Haar / HOG，相关算子会给出明确提示而不是
> `AttributeError`。

## OpenCV 5.0 算子目录

| 分类 | 数量 | 代表能力 |
| --- | --- | --- |
| 基础调整 | 18 | 加减乘除、位运算、归一化、伽马、LUT |
| 滤波与平滑 | 14 | 双边、方框、可分离卷积、锐化、金字塔、均值漂移 |
| 阈值与分割 | 19 | Otsu / 三角 / 自适应阈值、分水岭、GrabCut、K 均值、轮廓 |
| 边缘检测 | 10 | Canny、Sobel、Scharr、Laplacian、霍夫线 / 圆、LSD |
| 形态学 | 10 | 腐蚀膨胀、开闭、梯度、顶帽、黑帽、击中击不中 |
| 几何变换 | 13 | 仿射 / 透视、remap、极坐标、扩边、错切、裁剪、平铺 |
| 通道与色彩 | 13 | 颜色空间、通道混合、伪彩色、白平衡、色温、Alpha |
| 特征与分析 | 37 | SIFT / ORB / AKAZE / BRISK、匹配、直方图、矩、FFT / IFFT、PSNR / SSIM |
| 图像增强 | 17 | 非局部均值、TV-L1、修复、无缝克隆、HDR、去雾、Retinex |
| 视频与运动 | 10 | 光流、背景建模、CamShift、ECC、目标跟踪 |
| 立体与标定 | 6 | 棋盘格、畸变校正、立体视差 / 深度、单应、全景 |
| 深度学习 | 6 | YuNet 人脸（ONNX）、二维码 / 条码检测、DNN Blob、DNN 推理（预置 MobileNetV2）；Haar / HOG 仅 OpenCV 4.x |
| 交互处理 | 19 | ROI、线 / 矩形 / 圆 / 文本 / 箭头 / 水印、布局拼接 |
| 其他 | 3 | OpenCV 通用调用（任意 cv2 函数）、构建信息、输出预览（万能入口 / 出口） |

完整 id、说明与覆盖矩阵见 [docs/opencv_catalog.md](docs/opencv_catalog.md)。该文档由
`python scripts/gen_opencv_catalog_doc.py` 从注册表生成，新增算子后可重新运行同步。

## 工程结构

```
app/
  core/        图像容器、端口数据类型、算子组模型、DAG 模型、.mi 工程格式、图像 IO、ONNX 模型下载
  operators/   算子规范、自动扫描、内置算子、算子组
  engine/      异步执行引擎（快照 / 拓扑并行 / 下游递归重算 / 过期丢弃 / 算子组执行）
  graph/       QGraphicsView 节点画布、算子组容器、连线、命令、自动布局
  ui/          主窗口、工作区、算子库、预览、参数面板（含自定义控件）、替换算子、设置、主题
operators/     用户算子目录（自动扫描；算子组保存的算子写入 operators/custom/）
docs/          架构、用户手册、算子开发指南、算子组实现说明
packaging/     PyInstaller spec 与一键打包脚本
tests/         unittest 测试（无需 pytest）
```

## 开发与测试

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -t tests -p "test_*.py" -v
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
* 算子组内不能保存原始图像节点；已连线节点被拖入 / 拖出算子组时会自动改接到组端口上。

## 许可

MIT。
