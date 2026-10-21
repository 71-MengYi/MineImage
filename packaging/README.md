# packaging：PyInstaller 打包

本目录是 MineImage 的 onefile 打包配置与一键脚本。

## 文件

| 文件 | 用途 |
| --- | --- |
| `MineImage.spec` | PyInstaller onefile 配置：入口 `launcher.py`，收集内置算子源码、`assets/`、`docs/`、`operators/` 模板与 `LICENSE` |
| `launcher.py` | 打包入口，调用 `app.main:main` |
| `build_exe.ps1` / `build_exe.bat` | 一键脚本：可选安装 PyInstaller → 生成图标 → 构建 → 把 `docs/`、`examples/`、`operators/`、`sample_images/`、`README.md` 复制到 `dist/` |
| `make_icon.py` | 由 `assets/icon.svg` 生成 `.ico` / `.png` |
| `rthook_normalize_resources.py` | PyInstaller 运行时钩子：截断被文件系统/杀软补齐到块大小、尾部带 NUL 的 `.py` 资源，避免冻结后 `SyntaxError: source code string cannot contain null bytes` |

## 构建

推荐在仓库根目录：

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1
```

如果 PyInstaller 只装在本地目录（例如 `.tools`，不污染全局环境）：

```powershell
$env:PYTHONPATH = "$PWD\.tools"
python -m pip install --target .tools pyinstaller pillow   # 首次
powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1 -SkipInstall
# 或者直接：
$env:PYTHONPATH = "$PWD\.tools"
python -m PyInstaller --noconfirm --clean packaging/MineImage.spec
```

产物：`dist/MineImage.exe`（onefile，约 99 MiB）。首次启动需要解包，可能比源码
运行慢几秒。

## 运行期文件布局

* 配置：`dist/config/MineImage.ini`
* 日志：`dist/config/MineImage.log`
* 用户算子：`dist/operators/`（首次启动自动创建；启动日志会打印
  `算子库：N 个算子，M 个加载问题`）
* 文档 / 示例图像 / 演示工程：`dist/docs/`、`dist/sample_images/`、`dist/examples/`
  （由 `build_exe.ps1` 复制）
* 模型：`dist/models/`（首次运行人脸检测时自动下载 YuNet ONNX，或手动放入；
  可用 `MINEIMAGE_MODEL_DIR` 指向其它目录）

## 说明

* `MineImage.spec` 的 `datas` 只收集 `app/operators/builtin/*.py`，不会把开发机的
  `__pycache__` 打进 exe；内置算子由运行时 `OperatorRegistry` 从解包目录扫描。
* `runtime_hooks` 中的 `rthook_normalize_resources.py` 是健壮性补丁：正常系统上
  是 no-op，只在资源尾部出现 NUL 填充时截断。修改 spec 时请保留它。
* `dist/`、`build/`、`.tools/` 均已被 `.gitignore` 忽略，不要提交构建产物。
* **受限沙箱提示**：PyInstaller onefile 启动时必须先把资源解包到临时目录。若
  运行环境禁止子进程写入临时目录（例如某些 CI/沙箱），exe 会报
  `Failed to extract ...: fopen: Permission denied`。这不是打包内容损坏；可在
  普通桌面环境验证，或用 `--onedir` 构建做启动冒烟测试。
