"""在 Qt 线程池中下载 OpenCV ONNX 模型。

参数面板的「下载模型」按钮与设置对话框的模型页共用本模块：

* 下载在后台线程执行，不阻塞界面；
* 进度、成功、失败、取消通过 :class:`DownloadSignals` 发回 UI 线程；
* 下载完成后模型已通过 :func:`app.core.models.ensure_model` 校验并原子落盘；
* 控件销毁后迟到的信号会被安全忽略，不会触发 "Internal C++ object already
  deleted"。
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.core import models


class DownloadSignals(QObject):
    """后台下载任务发出的信号（必须由调用方持有，建议以控件为 parent）。"""

    progress = Signal(int, int)  # 已下载字节, 总字节（0 表示未知）
    succeeded = Signal(str)  # 模型本地路径
    failed = Signal(str)  # 中文错误信息
    cancelled = Signal()
    finished = Signal()  # 无论成功 / 失败 / 取消都会发出


class ModelDownloadTask(QRunnable):
    """下载一个模型；通过 :func:`start_model_download` 提交到线程池。"""

    def __init__(
        self,
        key: str,
        signals: DownloadSignals,
        *,
        timeout: float = models.DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__()
        self._key = str(key)
        self._signals = signals
        self._timeout = float(timeout)
        self._cancelled = threading.Event()
        self.setAutoDelete(False)

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def key(self) -> str:
        return self._key

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    # ------------------------------------------------------------------ 线程
    def run(self) -> None:  # noqa: D401 - QRunnable 接口
        try:
            path = models.ensure_model(
                self._key,
                auto=True,
                progress=self._on_progress,
                cancel=self._cancelled.is_set,
                timeout=self._timeout,
            )
        except models.ModelDownloadError as exc:
            if self._cancelled.is_set():
                self._emit(self._signals.cancelled)
            else:
                self._emit(self._signals.failed, str(exc))
        except models.ModelError as exc:
            self._emit(self._signals.failed, str(exc))
        except Exception as exc:  # noqa: BLE001 - 后台线程兜底，避免静默退出
            self._emit(self._signals.failed, f"模型下载失败：{exc}")
        else:
            self._emit(self._signals.succeeded, str(path))
        finally:
            self._emit(self._signals.finished)

    def _on_progress(self, downloaded: int, total: int) -> None:
        self._emit(self._signals.progress, int(downloaded), int(total))

    @staticmethod
    def _emit(signal: Signal, *args: object) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # 接收控件 / 信号对象已随窗口销毁，忽略这次迟到信号。
            pass


def start_model_download(
    key: str,
    signals: DownloadSignals,
    *,
    timeout: float = models.DEFAULT_TIMEOUT,
) -> ModelDownloadTask:
    """把下载任务提交到全局线程池并返回任务对象（调用方需持有引用）。"""

    task = ModelDownloadTask(key, signals, timeout=timeout)
    QThreadPool.globalInstance().start(task)
    return task
