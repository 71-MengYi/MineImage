"""SQLite 应用日志存储的单元测试（不依赖 Qt）。"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.config import default_log_db_path
from app.core.logstore import (
    LOG_LEVELS,
    LogFilter,
    LogStore,
    SQLiteLogHandler,
    format_time,
    normalize_level,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _temp_dir(prefix: str) -> Path:
    """在工作区内创建临时目录（不使用 tempfile，避免 Windows 沙箱 ACL 问题）。"""

    base = _REPO_ROOT / ".test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{prefix}-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    return path


class LogStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:  # noqa: N802
        self.temp = _temp_dir("logstore")
        self.db_path = self.temp / "nested" / "logs.db"
        self.store = LogStore(self.db_path, max_entries=1000)

    def tearDown(self) -> None:  # noqa: N802
        shutil.rmtree(self.temp, ignore_errors=True)

    # ------------------------------------------------------------------ 写入/读取
    def test_append_generates_ids_and_reads_newest_first(self) -> None:
        first = self.store.append("info", "第一条")
        second = self.store.append("ERROR", "第二条")
        third = self.store.append("CRITICAL", "第三条")

        self.assertTrue(first.entry_id.startswith("LOG-"))
        self.assertTrue(second.entry_id.startswith("ERR-"))
        self.assertTrue(third.entry_id.startswith("ERR-"))
        self.assertEqual(len({first.entry_id, second.entry_id, third.entry_id}), 3)
        self.assertGreater(first.seq, 0)
        self.assertLess(first.seq, second.seq)
        self.assertTrue(self.db_path.exists())

        entries = self.store.query(LogFilter(limit=None))
        self.assertEqual([entry.message for entry in entries], ["第三条", "第二条", "第一条"])
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(entries[0].level, "CRITICAL")

    def test_invalid_and_alias_levels_are_normalized(self) -> None:
        self.assertEqual(normalize_level("warn"), "WARNING")
        self.assertEqual(normalize_level("fatal"), "CRITICAL")
        self.assertEqual(normalize_level("verbose"), "INFO")
        entry = self.store.append("warn", "别名")
        self.assertEqual(entry.level, "WARNING")

    def test_full_text_contains_all_fields(self) -> None:
        entry = self.store.append(
            "ERROR",
            "图像读取失败",
            logger="app.core.imageio",
            detail="Traceback: ...",
            source="imageio.py:42",
            thread="worker-1",
        )
        text = entry.full_text()
        for expected in (
            entry.entry_id,
            "ERROR",
            "app.core.imageio",
            "imageio.py:42",
            "worker-1",
            "图像读取失败",
            "Traceback: ...",
        ):
            self.assertIn(expected, text)

    # ------------------------------------------------------------------ 筛选
    def test_level_filter(self) -> None:
        for level in LOG_LEVELS:
            self.store.append(level, f"level={level}")

        errors = self.store.query(LogFilter(levels=("ERROR", "CRITICAL")))
        self.assertEqual({entry.level for entry in errors}, {"ERROR", "CRITICAL"})
        self.assertEqual(len(errors), 2)
        self.assertEqual(self.store.query(LogFilter(levels=())).__len__(), 5)
        self.assertEqual(self.store.count(LogFilter(levels=("INFO",))), 1)

    def test_date_range_is_inclusive(self) -> None:
        self.store.append("INFO", "昨天", created_at=datetime(2026, 10, 21, 23, 59, 59, 999000))
        self.store.append("INFO", "今天一早", created_at=datetime(2026, 10, 22, 0, 0, 0, 1000))
        self.store.append("INFO", "明天", created_at=datetime(2026, 10, 23, 0, 0, 0, 1000))

        by_dates = LogFilter.by_dates(date(2026, 10, 22), date(2026, 10, 22))
        entries = self.store.query(by_dates)
        self.assertEqual([entry.message for entry in entries], ["今天一早"])

        explicit = LogFilter(
            start=datetime(2026, 10, 21, 23, 59, 59, 999000),
            end=datetime(2026, 10, 22, 0, 0, 0, 1000),
            limit=None,
        )
        self.assertEqual({entry.message for entry in self.store.query(explicit)}, {"昨天", "今天一早"})

    def test_keyword_matches_message_detail_logger_and_id(self) -> None:
        percent = self.store.append("INFO", "完成度 100%", detail="进度详情")
        self.store.append("INFO", "普通消息", detail="traceback: ValueError")
        underscore = self.store.append("INFO", "file_name.png")

        self.assertEqual(
            [entry.entry_id for entry in self.store.query(LogFilter(keyword="100%"))],
            [percent.entry_id],
        )
        self.assertEqual(
            [entry.entry_id for entry in self.store.query(LogFilter(keyword="traceback"))],
            [self.store.query(LogFilter(limit=None))[1].entry_id],
        )
        self.assertEqual(
            [entry.entry_id for entry in self.store.query(LogFilter(keyword="file_name"))],
            [underscore.entry_id],
        )
        # 下划线按字面匹配，不能当成 LIKE 通配符。
        self.assertEqual(self.store.query(LogFilter(keyword="fileXname")), [])
        self.assertEqual(
            [entry.entry_id for entry in self.store.query(LogFilter(keyword=percent.entry_id))],
            [percent.entry_id],
        )
        self.assertEqual(
            [entry.entry_id for entry in self.store.query(LogFilter(keyword="app.core."))],
            [],
        )

    def test_logger_filter_and_distinct_loggers(self) -> None:
        self.store.append("INFO", "a", logger="app.ui.workspace")
        self.store.append("INFO", "b", logger="app.engine.engine")
        self.store.append("ERROR", "c", logger="app.ui.workspace")

        self.assertEqual(
            self.store.loggers(), ["app.engine.engine", "app.ui.workspace"]
        )
        entries = self.store.query(LogFilter(logger="app.ui.workspace"))
        self.assertEqual({entry.message for entry in entries}, {"a", "c"})
        self.assertEqual(self.store.count(LogFilter(logger="missing")), 0)

    def test_limit_affects_rows_but_not_count(self) -> None:
        for index in range(5):
            self.store.append("INFO", f"消息 {index}", created_at=datetime(2026, 10, 22, 1, 0, index))
        entries = self.store.query(LogFilter(limit=2))
        self.assertEqual([entry.message for entry in entries], ["消息 4", "消息 3"])
        self.assertEqual(self.store.count(LogFilter(limit=2)), 5)

    def test_combined_filters(self) -> None:
        self.store.append("ERROR", "匹配", logger="app.core.x", created_at=datetime(2026, 10, 22, 10, 0))
        self.store.append("INFO", "匹配", logger="app.core.x", created_at=datetime(2026, 10, 22, 10, 0))
        self.store.append("ERROR", "不匹配", logger="app.core.y", created_at=datetime(2026, 10, 22, 10, 0))
        flt = LogFilter(
            levels=("ERROR",),
            start=datetime(2026, 10, 22, 0, 0),
            end=datetime(2026, 10, 22, 23, 59, 59),
            keyword="匹配",
            logger="app.core.x",
        )
        self.assertEqual([entry.message for entry in self.store.query(flt)], ["匹配"])

    # ------------------------------------------------------------------ logging 集成
    def test_logging_handler_captures_message_extra_and_traceback(self) -> None:
        logger = logging.getLogger(f"mineimage.test.{uuid.uuid4().hex[:8]}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = SQLiteLogHandler(self.store)
        logger.addHandler(handler)
        try:
            logger.info("处理 %s 完成", "图像", extra={"log_detail": "耗时 12ms"})
            try:
                raise ValueError("示例错误")
            except ValueError:
                logger.exception("算子执行失败")
            self.assertIsNotNone(handler.last_entry)
            self.assertEqual(handler.last_entry.level, "ERROR")
        finally:
            logger.removeHandler(handler)
            handler.close()

        entries = self.store.query(LogFilter(limit=None))
        self.assertEqual(len(entries), 2)
        error = entries[0]
        self.assertEqual(error.level, "ERROR")
        self.assertEqual(error.logger, logger.name)
        self.assertIn("ValueError: 示例错误", error.detail)
        self.assertIn("Traceback", error.detail)
        self.assertTrue(error.source.endswith(".py:" + error.source.split(":")[-1]))
        info = entries[1]
        self.assertEqual(info.message, "处理 图像 完成")
        self.assertIn("耗时 12ms", info.detail)

    def test_logging_handler_survives_store_failure(self) -> None:
        class BrokenStore(LogStore):
            def append_record(self, record):  # noqa: ANN001, ANN201
                raise RuntimeError("db down")

        broken = BrokenStore(self.temp / "broken.db")
        logger = logging.getLogger(f"mineimage.broken.{uuid.uuid4().hex[:8]}")
        logger.propagate = False
        handler = SQLiteLogHandler(broken)
        logger.addHandler(handler)
        try:
            # 故意让 handler 失败；关闭 logging 的“打印到 stderr”行为以保持测试输出干净。
            with patch.object(logging, "raiseExceptions", False):
                logger.warning("不应抛出")
        finally:
            logger.removeHandler(handler)
            handler.close()
        self.assertIsNone(handler.last_entry)

    # ------------------------------------------------------------------ 维护/导出
    def test_prune_keeps_newest_entries(self) -> None:
        for index in range(10):
            self.store.append("INFO", f"消息 {index}", created_at=datetime(2026, 10, 22, 1, 0, 0) + timedelta(seconds=index))
        removed = self.store.prune(5)
        self.assertEqual(removed, 5)
        entries = self.store.query(LogFilter(limit=None))
        self.assertEqual([entry.message for entry in entries], [f"消息 {index}" for index in range(9, 4, -1)])

    def test_clear_and_stats(self) -> None:
        self.store.append("INFO", "a", created_at=datetime(2026, 10, 22, 1, 0, 0))
        self.store.append("WARNING", "b", created_at=datetime(2026, 10, 22, 2, 0, 0))
        stats = self.store.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["first"], "2026-10-22 01:00:00.000")
        self.assertEqual(stats["last"], "2026-10-22 02:00:00.000")
        self.assertGreater(int(stats["size_bytes"]), 0)

        self.assertEqual(self.store.clear(), 2)
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.stats()["total"], 0)

    def test_export_csv_respects_filter_and_uses_bom(self) -> None:
        self.store.append("ERROR", "错误, 带逗号", detail="第一行\n第二行")
        self.store.append("INFO", "普通信息")
        target = self.temp / "export" / "logs.csv"
        exported = self.store.export_csv(target, LogFilter(levels=("ERROR",), limit=1))
        self.assertEqual(exported, 1)
        self.assertTrue(target.exists())

        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0][0], "日志ID")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][2], "ERROR")
        self.assertEqual(rows[1][4], "错误, 带逗号")
        self.assertIn("第二行", rows[1][5])
        self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_reopen_keeps_existing_rows(self) -> None:
        self.store.append("INFO", "持久化")
        reopened = LogStore(self.db_path)
        self.assertEqual([entry.message for entry in reopened.query(LogFilter())], ["持久化"])

    def test_concurrent_appends_are_safe(self) -> None:
        def worker(index: int) -> None:
            for second in range(25):
                self.store.append(
                    "INFO",
                    f"worker {index}-{second}",
                    created_at=datetime(2026, 10, 22, 1, 0, 0) + timedelta(seconds=index * 25 + second),
                )

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(4)))
        entries = self.store.query(LogFilter(limit=None))
        self.assertEqual(len(entries), 100)
        self.assertEqual(len({entry.entry_id for entry in entries}), 100)
        self.assertEqual({entry.thread for entry in entries}, {""})


class LogPathConfigTests(unittest.TestCase):
    def test_environment_override(self) -> None:
        target = _REPO_ROOT / ".test_tmp" / "custom-logs.db"
        with patch.dict(os.environ, {"MINEIMAGE_LOG_DB": str(target)}):
            self.assertEqual(default_log_db_path(), target)

    def test_default_path_sits_next_to_ini(self) -> None:
        with patch.dict(os.environ, {"MINEIMAGE_CONFIG": str(_REPO_ROOT / ".test_tmp" / "x.ini")}):
            path = default_log_db_path()
        self.assertEqual(path.name, "MineImage-logs.db")
        self.assertEqual(path.parent.name, ".test_tmp")


if __name__ == "__main__":
    unittest.main()
