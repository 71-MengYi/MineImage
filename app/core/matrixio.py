"""CSV / TSV / 文本矩阵读写。

MineImage 的源节点原本只接受图像；本模块让 ``.csv``、``.tsv``、``.tab``、
``.txt``、``.dat`` 这类数值矩阵文件也能作为源节点导入。导入结果统一转换成
**二维单通道** :class:`~app.core.types.ImageData`，因此可以直接使用 Imshow /
Imagesc / Mesh 查看，也能接入现有图像算子链。

解析规则
--------
* 自动识别逗号、制表符、分号、竖线与空白分隔符；Excel 导出的 ``sep=;`` 首行
  也会被识别并跳过；
* 文件编码依次尝试 UTF-8（含 BOM）、GBK、Latin-1；
* 跳过空行与以 ``#`` / ``%`` / ``//`` 开头的注释行；
* 首行没有任何数字单元格、且后续行存在数字时视为表头；首列全为非数字时视为
  行名（行列名只用于定位数据，不写入数组）；
* 空单元格记为 ``NaN``，支持 ``NaN`` / ``Inf`` / 科学计数法；分号或制表符
  分隔时 ``1,5`` 按小数 ``1.5`` 解析；
* 全整数矩阵使用 ``int32``，含小数或 NaN 时使用 ``float64``；
* 行列数不一致、非数字单元格、空文件等错误都抛 :class:`MatrixIOError`，
  消息包含文件名与行列号，便于用户直接修复。

写回时只支持二维单通道数组：``.csv`` 用逗号、``.tsv/.tab`` 用制表符、
``.txt/.dat`` 用空格分隔；多通道图像请导出为 ``.npy`` / PNG / TIFF。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.constants import MATRIX_SUFFIXES
from app.core.types import ImageData


class MatrixIOError(RuntimeError):
    """矩阵文件读写失败；消息为可直接展示给用户的中文。"""


#: 读取时依次尝试的编码；``latin-1`` 永远不会失败，作为最后兜底。
_ENCODINGS = ("utf-8-sig", "gbk", "latin-1")
#: 支持的分隔符；``None`` 表示按任意空白拆分。
_DELIMITERS = (",", "\t", ";", "|")
#: 各扩展名优先尝试的分隔符；其它分隔符仍会参与自动识别。
_DELIMITER_PREFERENCE = {
    ".csv": (",", ";", "\t", "|"),
    ".tsv": ("\t", ",", ";", "|"),
    ".tab": ("\t", ",", ";", "|"),
}
#: 注释行前缀。
_COMMENT_PREFIXES = ("#", "%", "//")
#: 被当作缺失值的文本（大小写不敏感）。
_MISSING_TOKENS = {"na", "n/a", "nan", "null", "none", "-", "--"}
#: 分隔符在提示文案里的中文名称。
_DELIMITER_LABELS = {",": "逗号", "\t": "制表符", ";": "分号", "|": "竖线", None: "空白"}
#: Excel 的无分隔符提示行：``sep=;``。
_SEP_DIRECTIVE = re.compile(r"^sep\s*=\s*([,;\t|])$", re.IGNORECASE)
#: 自动识别分隔符时最多采样的行数；避免大文件为每个候选分隔符重复全量解析。
_DETECT_SAMPLE_LINES = 64


@dataclass(frozen=True)
class MatrixData:
    """解析后的矩阵及其可选行列名；``array`` 始终为二维数组。"""

    array: np.ndarray
    column_labels: tuple[str, ...] = ()
    row_labels: tuple[str, ...] = ()
    delimiter: str | None = None
    has_header: bool = False
    source_name: str = ""

    def summary(self) -> str:
        """给状态栏 / 日志使用的一行摘要。"""

        height, width = self.array.shape
        parts = [
            f"{height} 行 × {width} 列",
            str(self.array.dtype),
            f"分隔符：{_DELIMITER_LABELS.get(self.delimiter, self.delimiter or '空白')}",
        ]
        if self.has_header:
            parts.append("含表头")
        if self.row_labels:
            parts.append("含行名")
        return " · ".join(parts)


def is_matrix_file(path: str | Path) -> bool:
    """判断路径是否为支持导入的文本矩阵文件（只看扩展名）。"""

    return Path(path).suffix.lower() in MATRIX_SUFFIXES


def read_matrix_file(path: str | Path, delimiter: str | None = None) -> MatrixData:
    """读取矩阵文件；``delimiter`` 为 ``None`` 时自动识别分隔符。"""

    file_path = Path(path)
    if not file_path.exists():
        raise MatrixIOError(f"矩阵文件不存在：{file_path}")
    if not is_matrix_file(file_path):
        supported = "、".join(sorted(MATRIX_SUFFIXES))
        raise MatrixIOError(
            f"不是支持的矩阵格式：{file_path.name}；支持 {supported}"
        )
    text = _read_text(file_path)
    return parse_matrix_text(
        text,
        name=file_path.name,
        suffix=file_path.suffix.lower(),
        delimiter=delimiter,
    )


def parse_matrix_text(
    text: str,
    *,
    name: str = "matrix",
    suffix: str = "",
    delimiter: str | None = None,
) -> MatrixData:
    """解析矩阵文本（导入与单元测试共用）。

    ``suffix`` 只影响分隔符的优先尝试顺序；``delimiter`` 显式给出时跳过自动
    识别，``None`` 表示自动识别。
    """

    lines = _content_lines(text.lstrip("\ufeff"))
    if not lines:
        raise MatrixIOError(f"矩阵文件为空或只有注释行：{name}")

    lines, directive = _take_sep_directive(lines)
    if delimiter is not None:
        chosen = delimiter
    elif directive is not None:
        chosen = directive
    else:
        chosen = _detect_delimiter(lines, suffix)
    if chosen is not None and chosen not in _DELIMITERS:
        raise MatrixIOError(f"不支持的分隔符：{chosen!r}；可选 , \\t ; | 或空白。")

    header_cells: tuple[str, ...] | None = None
    header_line_number = 0
    if _looks_like_header(lines, chosen):
        header_line_number, header_raw = lines[0]
        header_cells = _trim_trailing_empty(_split_line(header_raw, chosen))
        lines = lines[1:]

    parsed = _parse_data_rows(lines, chosen, name)
    if not parsed:
        raise MatrixIOError(f"矩阵文件没有数据行：{name}")

    widths = {len(row) for _line, row in parsed}
    if len(widths) != 1:
        expected = len(parsed[0][1])
        for line_number, row in parsed:
            if len(row) != expected:
                raise MatrixIOError(
                    f"{name}第 {line_number} 行有 {len(row)} 列，"
                    f"与前面的 {expected} 列不一致；请检查分隔符或补齐缺失单元格。"
                )
        raise MatrixIOError(f"{name}的数据行列数不一致；请检查分隔符或补齐缺失单元格。")

    full_width = len(parsed[0][1])
    data_rows = parsed
    row_labels: tuple[str, ...] = ()
    if (
        full_width >= 2
        and any(row[0].strip() for _line, row in parsed)
        and all(not _cell_is_number(row[0], chosen) for _line, row in parsed)
    ):
        row_labels = tuple(row[0].strip() for _line, row in parsed)
        data_rows = [(line, row[1:]) for line, row in parsed]
        if header_cells is not None and len(header_cells) == full_width:
            header_cells = header_cells[1:]

    column_count = len(data_rows[0][1])
    if column_count == 0:
        raise MatrixIOError(f"{name}没有可识别的数据列；请检查分隔符或文件内容。")

    column_labels = _column_labels(header_cells, header_line_number, column_count, name)
    values = _to_float_array(data_rows, column_count, chosen, name)
    array = _compact_dtype(values)
    return MatrixData(
        array=array,
        column_labels=column_labels,
        row_labels=row_labels,
        delimiter=chosen,
        has_header=header_cells is not None,
        source_name=name,
    )


def write_matrix_file(
    path: str | Path,
    image: ImageData | np.ndarray,
    delimiter: str | None = None,
) -> None:
    """把二维单通道数组写为 CSV / 文本矩阵。"""

    file_path = Path(path)
    suffix = file_path.suffix.lower() or ".csv"
    if suffix not in MATRIX_SUFFIXES:
        supported = "、".join(sorted(MATRIX_SUFFIXES))
        raise MatrixIOError(f"不支持的矩阵导出格式：{suffix or '（无扩展名）'}；支持 {supported}")

    array = image.array if isinstance(image, ImageData) else np.asarray(image)
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise MatrixIOError(
            "矩阵文件只支持二维单通道数据；彩色 / 多通道图像请导出为 "
            ".npy、PNG 或 TIFF。"
        )
    if array.size == 0:
        raise MatrixIOError("当前结果为空数组，无法导出矩阵文件。")

    chosen = delimiter
    if chosen is None:
        if suffix in (".tsv", ".tab"):
            chosen = "\t"
        elif suffix == ".csv":
            chosen = ","
        else:
            chosen = " "
    if chosen not in (",", "\t", ";", "|", " "):
        raise MatrixIOError(f"不支持的分隔符：{chosen!r}；可选 , \\t ; | 或空格。")

    if array.dtype.kind == "b":
        array = array.astype(np.uint8)
    fmt = "%d" if array.dtype.kind in "iu" else "%.15g"
    try:
        np.savetxt(str(file_path), array, delimiter=chosen, fmt=fmt)
    except (OSError, ValueError) as exc:
        raise MatrixIOError(f"无法写入矩阵文件：{exc}") from exc


# ---------------------------------------------------------------------------
# 读取与预处理
# ---------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MatrixIOError(f"无法读取矩阵文件：{exc}") from exc
    if b"\x00" in raw:
        raise MatrixIOError(
            f"不是文本矩阵文件（包含二进制内容）：{path.name}；"
            "数值数组请使用 .npy 保存。"
        )
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise MatrixIOError(f"无法识别矩阵文件编码：{path.name}；请转存为 UTF-8。")  # pragma: no cover


def _content_lines(text: str) -> list[tuple[int, str]]:
    """返回 ``(原始行号, 行文本)``，跳过空行与注释行。"""

    lines: list[tuple[int, str]] = []
    for index, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        lines.append((index, raw.rstrip("\r\n")))
    return lines


def _take_sep_directive(
    lines: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], str | None]:
    """识别并移除 Excel 的 ``sep=;`` 提示行。"""

    if not lines:
        return lines, None
    match = _SEP_DIRECTIVE.match(lines[0][1].strip())
    if match is None:
        return lines, None
    return lines[1:], match.group(1)


def _detect_delimiter(lines: list[tuple[int, str]], suffix: str) -> str | None:
    """按“是否多列 → 列数一致性 → 数字占比 → 列数”选出分隔符，``None`` 表示空白。"""

    preferred = _DELIMITER_PREFERENCE.get(suffix, _DELIMITERS)
    order: list[str | None] = list(preferred)
    order.extend(item for item in _DELIMITERS if item not in order)
    order.append(None)

    best_score: tuple[float, float, float, int] | None = None
    best: str | None = None
    sample = lines[:_DETECT_SAMPLE_LINES]
    for candidate in order:
        score = _delimiter_score(sample, candidate)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = candidate
    return best


def _delimiter_score(
    lines: list[tuple[int, str]], delimiter: str | None
) -> tuple[float, float, float, int] | None:
    rows: list[tuple[str, ...]] = []
    for _line_number, raw in lines:
        try:
            rows.append(_split_line(raw, delimiter))
        except csv.Error:
            return None
    if not rows:
        return None

    widths = [len(row) for row in rows]
    modal = max(set(widths), key=lambda value: (widths.count(value), value))
    if delimiter is not None and modal < 2:
        return None
    consistency = widths.count(modal) / float(len(widths))
    total = 0
    numeric = 0
    for row in rows:
        for cell in row:
            total += 1
            if _cell_is_number(cell, delimiter):
                numeric += 1
    ratio = numeric / float(total) if total else 0.0
    # 多列解析优先于单列（空白兜底），再比较列数一致性与数字占比。
    return (1.0 if modal >= 2 else 0.0, consistency, ratio, modal)


def _split_line(raw: str, delimiter: str | None) -> tuple[str, ...]:
    if delimiter is None:
        return tuple(raw.split())
    reader = csv.reader([raw], delimiter=delimiter)
    try:
        return tuple(next(reader))
    except StopIteration:  # pragma: no cover - csv.reader 对单行总会返回一行
        return ()


def _trim_trailing_empty(cells: tuple[str, ...]) -> tuple[str, ...]:
    end = len(cells)
    while end > 0 and not cells[end - 1].strip():
        end -= 1
    return cells[:end]


def _parse_data_rows(
    lines: list[tuple[int, str]], delimiter: str | None, name: str
) -> list[tuple[int, tuple[str, ...]]]:
    parsed: list[tuple[int, tuple[str, ...]]] = []
    for line_number, raw in lines:
        try:
            cells = _split_line(raw, delimiter)
        except csv.Error as exc:
            raise MatrixIOError(
                f"{name}第 {line_number} 行无法按"
                f"{_DELIMITER_LABELS.get(delimiter, '空白')}拆分：{exc}"
            ) from exc
        cells = _trim_trailing_empty(cells)
        if not cells:
            continue
        parsed.append((line_number, cells))
    return parsed


def _looks_like_header(lines: list[tuple[int, str]], delimiter: str | None) -> bool:
    if len(lines) < 2:
        return False
    try:
        first = _trim_trailing_empty(_split_line(lines[0][1], delimiter))
    except csv.Error:
        return False
    if not first or any(_cell_is_number(cell, delimiter) for cell in first):
        return False
    for _line_number, raw in lines[1:]:
        try:
            row = _trim_trailing_empty(_split_line(raw, delimiter))
        except csv.Error:
            return False
        if row and _row_is_data(row, delimiter):
            return True
    return False


def _row_is_data(row: tuple[str, ...], delimiter: str | None) -> bool:
    """判断一行是否包含数据；允许首列是行名（非数字）。"""

    if _row_is_numeric(row, delimiter):
        return True
    if len(row) >= 2 and row[0].strip() and not _cell_is_number(row[0], delimiter):
        return _row_is_numeric(row[1:], delimiter)
    return False


def _row_is_numeric(row: tuple[str, ...], delimiter: str | None) -> bool:
    seen = False
    for cell in row:
        if _cell_is_number(cell, delimiter):
            seen = True
        elif cell.strip():
            return False
    return seen


def _cell_is_number(cell: str, delimiter: str | None) -> bool:
    text = cell.strip()
    if not text:
        return False
    try:
        _cell_to_float(text, delimiter)
    except ValueError:
        return False
    return True


def _cell_to_float(text: str, delimiter: str | None) -> float:
    """把一个单元格转成 float；分号/制表符分隔时接受小数逗号。"""

    value = text.strip()
    if not value:
        return float("nan")
    if value.lower() in _MISSING_TOKENS:
        return float("nan")
    if delimiter != "," and "," in value and "." not in value:
        value = value.replace(",", ".")
    return float(value)


def _column_labels(
    header_cells: tuple[str, ...] | None,
    line_number: int,
    column_count: int,
    name: str,
) -> tuple[str, ...]:
    if header_cells is None:
        return ()
    cells = list(header_cells)
    if len(cells) == column_count:
        labels = cells
    elif len(cells) < column_count:
        labels = cells + [f"C{index + 1}" for index in range(len(cells), column_count)]
    else:
        raise MatrixIOError(
            f"{name}第 {line_number} 行的表头有 {len(cells)} 列，"
            f"多于数据的 {column_count} 列；请检查表头与数据是否对齐。"
        )
    return tuple(label.strip() or f"C{index + 1}" for index, label in enumerate(labels))


def _to_float_array(
    rows: list[tuple[int, tuple[str, ...]]],
    column_count: int,
    delimiter: str | None,
    name: str,
) -> np.ndarray:
    values = np.empty((len(rows), column_count), dtype=np.float64)
    for row_index, (line_number, row) in enumerate(rows):
        for column_index in range(column_count):
            cell = row[column_index] if column_index < len(row) else ""
            try:
                values[row_index, column_index] = _cell_to_float(cell, delimiter)
            except ValueError as exc:
                raise MatrixIOError(
                    f"{name}第 {line_number} 行第 {column_index + 1} 列的值不是数字："
                    f"{cell.strip()!r}；请检查分隔符或数据格式。"
                ) from exc
    return values


def _compact_dtype(values: np.ndarray) -> np.ndarray:
    """全整数且落在 int32 范围内时压缩为 int32，其余保持 float64。"""

    if values.size and np.all(np.isfinite(values)) and np.all(values == np.floor(values)):
        info = np.iinfo(np.int32)
        low = float(values.min())
        high = float(values.max())
        if info.min <= low and high <= info.max:
            return values.astype(np.int32)
    return values
