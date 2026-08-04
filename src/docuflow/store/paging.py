"""One page of rows, and the numbers needed to draw a pager.

Every list in this system grows without bound - documents, ledger records, a
vendor master file that arrives as a CSV of several thousand. Returning all of
them and letting the browser cope works right up until it does not, and the
failure is silent: the table renders, it is simply missing the rows nobody
scrolled to.

So the lists are paged at the source. The envelope carries what a pager needs
and nothing else, and `total` is a real count rather than a guess from the page
size, because "showing 10 of 10" when there are 340 is worse than no number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_PER_PAGE = 10
MAX_PER_PAGE = 200


def clamp_page(page: int, per_page: int) -> tuple[int, int, int]:
    """Sanitise the requested page, returning (page, per_page, offset)."""
    per_page = max(1, min(int(per_page or DEFAULT_PER_PAGE), MAX_PER_PAGE))
    page = max(1, int(page or 1))
    return page, per_page, (page - 1) * per_page


@dataclass(frozen=True)
class Page:
    rows: Sequence[Any]
    total: int
    page: int
    per_page: int

    @property
    def pages(self) -> int:
        """At least one, so an empty list is "page 1 of 1" rather than 1 of 0."""
        return max(1, -(-self.total // self.per_page))

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": list(self.rows),
            "total": self.total,
            "page": self.page,
            "per_page": self.per_page,
            "pages": self.pages,
        }


def paginate(rows: Sequence[Any], total: int, page: int, per_page: int) -> dict[str, Any]:
    return Page(rows=rows, total=total, page=page, per_page=per_page).to_json()


def slice_all(rows: Sequence[Any], page: int, per_page: int) -> dict[str, Any]:
    """Page a list already in memory.

    For the master files, which are read whole anyway because vendor matching
    compares against every row. Paging them in SQL would save nothing and would
    make the two code paths disagree about ordering.
    """
    page, per_page, offset = clamp_page(page, per_page)
    return paginate(rows[offset : offset + per_page], len(rows), page, per_page)
