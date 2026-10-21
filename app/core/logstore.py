"""应用日志的 SQLite 存储与 logging 处理器。

日志写入 app_logs 表，供「帮助 → 日志」查看器筛选、复制和导出。设计要点：

* 每条记录有可读的日志 ID：ERROR/CRITICAL 以 ERR- 开头，其余以 LOG- 开头；
* 时间以本地时间 YYYY-MM-DD HH:MM:SS.mmm 存储，字符串排序即时间排序；
* 每次读写使用独立连接并启用 WAL，允许 UI 查看日志时后台线程继续写入；
* 写入量很小，因此不做长连接；出现任何数据库错误都由调用方决定如何降级；
* 数据库只追加写，启动时按 max_entries 自动清理最旧的记录，避免无限增长。
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import threading
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterator

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
"""日志查看器展示的等级顺序。"""

_LEVEL_ALIASES = {"WARN": "WARNING", "ERR": "ERROR", "FATAL": "CRITICAL"}
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
_MAX_KEYWORD_LENGTH = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_logs (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id   TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    level      TEXT NOT NULL,
    logger     TEXT NOT NULL,
    message    TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL DEFAULT '',
    thread     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_app_logs_created_at ON app_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_app_logs_level ON app_logs(level);
CREATE INDEX IF NOT EXISTS idx_app_logs_logger ON app_logs(logger);
"""

_ENTRY_COLUMNS = "seq, entry_id, created_at, level, logger, message, detail, source, thread"


def normalize_level(level: str) -> str:
    """把等级规范成 LOG_LEVELS 中的大写名称；未知值按 INFO 处理。"""

    text = str(level or "").strip().upper()
    text = _LEVEL_ALIASES.get(text, text)
    return text if text in LOG_LEVELS else "INFO"


def format_time(value: datetime) -> str:
    """数据库时间格式（毫秒精度，便于字符串比较）。"""

    return value.strftime(_TIME_FORMAT)[:-3]


def make_entry_id(level: str, created_at: datetime) -> str:
    """生成可读、可搜索的日志 ID，例如 ERR-20261022-011214-A1B2C3D4。"""

    prefix = "ERR" if normalize_level(level) in ("ERROR", "CRITICAL") else "LOG"
    # 8 位随机十六进制 + 秒级时间戳：同一秒内即使有大量错误也很少碰撞。
    return f"{prefix}-{created_at:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8].upper()}"


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass(frozen=True)
class LogEntry:
    """一条日志记录。"""

    seq: int
    entry_id: str
    created_at: str
    level: str
    logger: str
    message: str
    detail: str = ""
    source: str = ""
    thread: str = ""

    @property
    def is_error(self) -> bool:
        return self.level in ("ERROR", "CRITICAL")

    @property
    def short_time(self) -> str:
        """精确到秒的时间，用于表格。"""

        return self.created_at[:19]

    def full_text(self) -> str:
        """用于复制 / 导出的纯文本全文。"""

        lines = [
            f"日志 ID：{self.entry_id}",
            f"时间：{self.created_at}",
            f"等级：{self.level}",
            f"模块：{self.logger}",
            f"来源：{self.source or '-'}",
            f"线程：{self.thread or '-'}",
            "",
            self.message,
        ]
        if self.detail:
            lines.extend(["", "─" * 64, "", self.detail])
        return "\n".join(lines)


@dataclass(frozen=True)
class LogFilter:
    """日志查询条件；字段为默认值时表示不限制。"""

    levels: tuple[str, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    keyword: str = ""
    logger: str = ""
    limit: int | None = 1000

    def __post_init__(self) -> None:
        levels = tuple(
            dict.fromkeys(
                normalize_level(level)
                for level in self.levels
                if str(level or "").strip()
            )
        )
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "keyword", str(self.keyword or "").strip()[:_MAX_KEYWORD_LENGTH])
        object.__setattr__(self, "logger", str(self.logger or "").strip())

    @classmethod
    def by_dates(
        cls,
        start: date | None = None,
        end: date | None = None,
        **kwargs: object,
    ) -> "LogFilter":
        """按“整天”构造日期范围（含起止两天）。"""

        start_dt = datetime.combine(start, time.min) if start is not None else None
        end_dt = datetime.combine(end, time.max) if end is not None else None
        return cls(start=start_dt, end=end_dt, **kwargs)  # type: ignore[arg-type]


def _build_where(flt: LogFilter) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if flt.levels:
        placeholders = ", ".join("?" for _ in flt.levels)
        clauses.append(f"level IN ({placeholders})")
        params.extend(flt.levels)
    if flt.start is not None:
        clauses.append("created_at >= ?")
        params.append(format_time(flt.start))
    if flt.end is not None:
        clauses.append("created_at <= ?")
        params.append(format_time(flt.end))
    if flt.logger:
        clauses.append("logger = ?")
        params.append(flt.logger)
    if flt.keyword:
        pattern = f"%{_escape_like(flt.keyword)}%"
        clauses.append(
            "(message LIKE ? ESCAPE '\\' OR detail LIKE ? ESCAPE '\\' "
            "OR logger LIKE ? ESCAPE '\\' OR entry_id LIKE ? ESCAPE '\\' "
            "OR source LIKE ? ESCAPE '\\')"
        )
        params.extend([pattern] * 5)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


class LogStore:
    """SQLite 日志仓库；线程安全，可在多个线程中同时调用。"""

    def __init__(self, path: str | Path, *, max_entries: int = 50000) -> None:
        self.path = Path(path)
        self.max_entries = max(0, int(max_entries))
        self._lock = threading.RLock()
        self._ensure_schema()
        self.prune()

    # ------------------------------------------------------------------ 连接
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(_SCHEMA)

    # ------------------------------------------------------------------ 写入
    def append(
        self,
        level: str,
        message: str,
        *,
        logger: str = "mineimage",
        detail: str = "",
        source: str = "",
        thread: str = "",
        created_at: datetime | None = None,
    ) -> LogEntry:
        """追加一条日志并返回落库后的记录。"""

        level_name = normalize_level(level)
        created = created_at or datetime.now()
        entry_id = make_entry_id(level_name, created)
        message_text = str(message)
        detail_text = str(detail or "")
        source_text = str(source or "")
        thread_text = str(thread or "")
        logger_text = str(logger or "mineimage")
        created_text = format_time(created)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO app_logs "
                "(entry_id, created_at, level, logger, message, detail, source, thread) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    created_text,
                    level_name,
                    logger_text,
                    message_text,
                    detail_text,
                    source_text,
                    thread_text,
                ),
            )
            seq = int(cursor.lastrowid or 0)
        return LogEntry(
            seq=seq,
            entry_id=entry_id,
            created_at=created_text,
            level=level_name,
            logger=logger_text,
            message=message_text,
            detail=detail_text,
            source=source_text,
            thread=thread_text,
        )

    def append_record(self, record: logging.LogRecord) -> LogEntry:
        """把 logging.LogRecord 转成日志记录。

        支持 extra={"log_detail": "..."} 携带详细信息；异常堆栈会被
        自动收进 detail，无需在调用处再格式化 traceback。
        """

        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - 格式串与参数不匹配时仍要写日志
            message = str(record.msg)
        detail_parts: list[str] = []
        extra_detail = getattr(record, "log_detail", "")
        if extra_detail:
            detail_parts.append(str(extra_detail).rstrip())
        if record.exc_info:
            detail_parts.append(
                "".join(traceback.format_exception(*record.exc_info)).rstrip()
            )
        elif record.exc_text:
            detail_parts.append(str(record.exc_text).rstrip())
        stack_info = getattr(record, "stack_info", None)
        if stack_info:
            detail_parts.append(str(stack_info).rstrip())
        source = f"{record.filename}:{record.lineno}" if record.filename else ""
        return self.append(
            record.levelname,
            message,
            logger=record.name,
            detail="\n\n".join(part for part in detail_parts if part),
            source=source,
            thread=record.threadName or "",
            created_at=datetime.fromtimestamp(record.created),
        )

    # ------------------------------------------------------------------ 查询
    def query(self, flt: LogFilter | None = None) -> list[LogEntry]:
        criteria = flt or LogFilter(limit=None)
        where, params = _build_where(criteria)
        sql = f"SELECT {_ENTRY_COLUMNS} FROM app_logs{where} ORDER BY seq DESC"
        limit = criteria.limit
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def count(self, flt: LogFilter | None = None) -> int:
        criteria = flt or LogFilter(limit=None)
        where, params = _build_where(criteria)
        with self._lock, self._connection() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM app_logs{where}", params
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def loggers(self) -> list[str]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT logger FROM app_logs ORDER BY logger"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def stats(self) -> dict[str, object]:
        """数据库摘要，供查看器底部状态栏显示。"""

        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM app_logs"
            ).fetchone()
        size = self.size_bytes()
        return {
            "total": int(row[0]) if row is not None else 0,
            "first": str(row[1] or "") if row is not None else "",
            "last": str(row[2] or "") if row is not None else "",
            "size_bytes": size,
        }

    def size_bytes(self) -> int:
        """数据库文件大小（含 WAL 旁路文件），不存在时返回 0。"""

        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
        return total

    # ------------------------------------------------------------------ 维护
    def prune(self, max_entries: int | None = None) -> int:
        """只保留最新的 max_entries 条记录，返回删除条数。"""

        limit = self.max_entries if max_entries is None else max(0, int(max_entries))
        if limit <= 0:
            return 0
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM app_logs WHERE seq NOT IN "
                "(SELECT seq FROM app_logs ORDER BY seq DESC LIMIT ?)",
                (limit,),
            )
            return int(cursor.rowcount if cursor.rowcount >= 0 else 0)

    def clear(self) -> int:
        """清空全部日志，返回删除条数。"""

        with self._lock, self._connection() as connection:
            cursor = connection.execute("DELETE FROM app_logs")
            deleted = int(cursor.rowcount if cursor.rowcount >= 0 else 0)
        return deleted

    def close(self) -> None:
        """兼容性接口：当前实现按需开关连接，无需常驻资源。"""

        return None

    # ------------------------------------------------------------------ 导出
    def export_csv(self, path: str | Path, flt: LogFilter | None = None) -> int:
        """把符合条件的全部日志导出为 CSV（UTF-8 BOM，Excel 可直接打开）。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        criteria = flt or LogFilter(limit=None)
        if criteria.limit is not None:
            criteria = replace(criteria, limit=None)
        entries = self.query(criteria)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["日志ID", "时间", "等级", "模块", "消息", "详细信息", "来源", "线程"]
            )
            for entry in entries:
                writer.writerow(
                    [
                        entry.entry_id,
                        entry.created_at,
                        entry.level,
                        entry.logger,
                        entry.message,
                        entry.detail,
                        entry.source,
                        entry.thread,
                    ]
                )
        return len(entries)

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LogEntry:
        return LogEntry(
            seq=int(row["seq"]),
            entry_id=str(row["entry_id"]),
            created_at=str(row["created_at"]),
            level=str(row["level"]),
            logger=str(row["logger"]),
            message=str(row["message"]),
            detail=str(row["detail"] or ""),
            source=str(row["source"] or ""),
            thread=str(row["thread"] or ""),
        )


class SQLiteLogHandler(logging.Handler):
    """把 logging 记录写入 LogStore 的处理器。

    日志写入失败时不会影响业务代码（走 handleError），避免“日志系统
    自身故障导致应用崩溃”。
    """

    def __init__(self, store: LogStore, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.store = store
        self._last_lock = threading.Lock()
        self._last_entry: LogEntry | None = None

    @property
    def last_entry(self) -> LogEntry | None:
        """最近一条成功落库的记录（用于异常对话框展示日志 ID）。"""

        with self._last_lock:
            return self._last_entry

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = self.store.append_record(record)
        except Exception:  # noqa: BLE001 - 日志失败不能冒泡到业务代码
            self.handleError(record)
            return
        with self._last_lock:
            self._last_entry = entry
