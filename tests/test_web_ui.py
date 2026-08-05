"""The web UI, in a real browser.

`web/app.js` is 2,053 lines and had no test touching it. Every UI bug found so
far came from there, and 753 Python tests caught none of them - because none of
them is about Python. They are about a stylesheet that pins one element to a
hard-coded height of another, a ternary that paints three severities in two
colours, and a badge counting something the table below it does not.

So this file is not general coverage. Each test is one bug that shipped, put
back in front of a browser, so that reintroducing it turns something red here
rather than in front of a user.

Run with `-m ui`. They need a browser (`python -m playwright install chromium`)
and are skipped when one is not installed, so the ordinary suite stays fast.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("playwright", reason="pip install pytest-playwright")
from playwright.sync_api import Page, sync_playwright  # noqa: E402

pytestmark = pytest.mark.ui


# --------------------------------------------------------------------------- #
# A real server, because half of what is being tested is HTTP headers and the
# other half is layout that only a browser computes.
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory) -> Iterator[str]:
    import uvicorn

    from docuflow import api

    port = _free_port()
    config = uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.skip("the API server did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser installed is a skip
            pytest.skip(f"no chromium: {exc}")
        yield launched
        launched.close()


@pytest.fixture
def page(browser, live_server) -> Iterator[Page]:
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    opened = context.new_page()
    opened.goto(live_server, wait_until="networkidle")
    yield opened
    context.close()


def _tab(page: Page, name: str) -> None:
    page.get_by_role("button", name=name).first.click()
    page.wait_for_timeout(400)


# --------------------------------------------------------------------------- #
# The bugs
# --------------------------------------------------------------------------- #


GAP = (
    "() => {const t=document.querySelector('.topbar').getBoundingClientRect();"
    "const n=document.querySelector('.tabs').getBoundingClientRect();"
    "return +(n.top - t.bottom).toFixed(1);}"
)


def _scroll_down(page: Page, to: int = 600) -> int:
    """Scroll, and prove the page moved.

    The first version of this test scrolled a page that could not scroll -
    viewport 900px, content 900px - so `scrollY` stayed 0, sticky positioning
    never engaged, and the assertion measured a layout at rest. It passed with
    the bug put back, which is worse than having no test: it reported that a
    shipped bug could not happen.
    """
    page.evaluate(f"window.scrollTo(0, {to})")
    page.wait_for_timeout(250)
    moved = page.evaluate("window.scrollY")
    assert moved > 50, (
        f"the page did not scroll (scrollY={moved}); sticky positioning never "
        "engaged and this test would pass without measuring anything"
    )
    return moved


class TestTheTabBarStaysBelowTheHeader:
    """`.tabs` was pinned at a hard-coded `top: 63px`.

    The header renders about 110px once the model names wrap, so on scroll
    roughly 47px of the tab bar slid underneath it. The number had to equal
    something it could not see, and did not.

    Every test here scrolls a genuinely scrollable page, because at rest the
    bug is invisible - both layouts look identical until sticky engages.
    """

    def test_the_tabs_sit_directly_under_the_header_at_rest(self, page: Page) -> None:
        assert abs(page.evaluate(GAP)) < 1.5

    def test_they_still_do_once_the_page_is_scrolled(self, page: Page) -> None:
        # A tall view, so there is something to scroll past.
        _tab(page, "Settings")
        _scroll_down(page)
        gap = page.evaluate(GAP)
        assert abs(gap) < 1.5, (
            f"scrolled, the tab bar is {gap}px out of place - a negative number "
            "means it slid underneath the header, which is the bug"
        )

    def test_it_holds_when_the_header_grows(self, page: Page) -> None:
        """Narrow enough that the model pickers wrap and the header gets taller.

        This is the case the hard-coded 63px could never have handled: the
        offset had to equal a height that changes with the window.
        """
        _tab(page, "Settings")
        page.set_viewport_size({"width": 720, "height": 700})
        page.wait_for_timeout(350)
        header = page.evaluate(
            "() => Math.round(document.querySelector('.topbar').getBoundingClientRect().height)"
        )
        assert header > 63, (
            f"the header is {header}px here, so this window does not exercise the bug"
        )
        _scroll_down(page)
        gap = page.evaluate(GAP)
        assert abs(gap) < 1.5, f"header {header}px tall, tab bar {gap}px out of place"


class TestControlsOnOneRowAreOneHeight:
    """The pickers were 34.6px, the threshold pill 29.8, the buttons 38.1 and
    the queue filter 19 - a bare `<select>` with native chrome beside custom
    buttons. Close enough to read as a rendering fault rather than a choice.
    """

    def test_the_header_controls_match(self, page: Page) -> None:
        heights = page.evaluate(
            "() => [...document.querySelectorAll('.picker, .engine')]"
            ".map(e => Math.round(e.getBoundingClientRect().height))"
        )
        assert heights, "no header controls found"
        assert len(set(heights)) == 1, f"header controls are {sorted(set(heights))}px"

    def test_buttons_and_selects_in_a_card_head_match(self, page: Page) -> None:
        _tab(page, "Review queue")
        heights = page.evaluate(
            "() => [...document.querySelectorAll('#view-queue .filters > *')]"
            ".map(e => Math.round(e.getBoundingClientRect().height))"
        )
        assert heights, "the queue filter row is empty"
        assert len(set(heights)) == 1, f"the filter row is {sorted(set(heights))}px"


class TestTheQueueAgreesWithItsOwnBadge:
    """The badge counted `needs_review` while the table below defaulted to
    every decision, so a badge reading 5 opened onto a list of 10.
    """

    def test_the_queue_opens_filtered_to_what_needs_a_person(self, page: Page) -> None:
        _tab(page, "Review queue")
        assert page.locator("#queue-filter").input_value() == "needs_review"

    def test_the_badge_and_the_table_report_the_same_number(self, page: Page) -> None:
        _tab(page, "Review queue")
        page.wait_for_timeout(600)
        badge = int(page.locator("#queue-badge").inner_text().strip() or 0)
        rows = page.evaluate(
            "() => document.querySelectorAll('#queue-table tbody tr td[colspan]').length"
            " ? 0 : document.querySelectorAll('#queue-table tbody tr').length"
        )
        assert rows == badge, f"badge says {badge}, the table shows {rows}"

    def test_an_empty_queue_does_not_claim_nothing_was_processed(self, page: Page) -> None:
        """Filtered to needs_review, empty means "nothing waiting on you"."""
        _tab(page, "Review queue")
        page.wait_for_timeout(600)
        empty = page.locator("#queue-table .empty-state")
        if empty.count():
            text = empty.inner_text().lower()
            assert "no documents yet" not in text, (
                "an empty *filtered* queue said nothing was processed, which reads "
                "as though the run was lost"
            )


class TestTheThreeSeveritiesLookDifferent:
    """`severity === 'blocking' ? 'bad' : 'warn'` painted every info note in
    the warning colour - the one colour that says something is wrong, on the
    one severity that means nothing is. It contradicted the project's own
    invariant that the three must behave, and read, differently.
    """

    def test_each_severity_maps_to_its_own_chip(self, page: Page) -> None:
        """The bug was here, in the mapping, not in the stylesheet.

        `severity === 'blocking' ? 'bad' : 'warn'` gave `info` the warning
        chip. Checking only that the three CSS classes look different would
        have passed throughout - the classes were fine, nothing ever asked for
        the third one.
        """
        chips = page.evaluate(
            "() => ['blocking','warning','info'].map(s => severityChip(s))"
        )
        assert len(set(chips)) == 3, f"severities map to {chips}; info must not share"
        assert chips[0] != chips[2], "blocking and info must not look the same"
        assert chips[1] != chips[2], (
            "info was painted in the warning colour - the one colour that says "
            "something is wrong, on the one severity that means nothing is"
        )

    def test_each_severity_gets_its_own_colour(self, page: Page) -> None:
        colours = page.evaluate(
            """() => {
              const probe = document.createElement('div');
              probe.style.position = 'absolute'; probe.style.visibility = 'hidden';
              probe.innerHTML = ['bad','warn','info']
                .map(k => `<span class="chip ${k}">x</span>`).join('');
              document.body.append(probe);
              const out = [...probe.children].map(el => {
                const s = getComputedStyle(el);
                return s.color + '|' + s.backgroundColor;
              });
              probe.remove();
              return out;
            }"""
        )
        assert len(set(colours)) == 3, f"blocking/warning/info share styling: {colours}"

    def test_a_finding_carries_its_severity_on_a_rail_too(self, page: Page) -> None:
        """So they stay apart when the wash is nearly white, or to an eye that
        cannot separate amber from red."""
        rails = page.evaluate(
            """() => {
              const probe = document.createElement('div');
              probe.style.position = 'absolute'; probe.style.visibility = 'hidden';
              probe.innerHTML = ['blocking','warning','info']
                .map(k => `<div class="finding ${k}">x</div>`).join('');
              document.body.append(probe);
              const out = [...probe.children].map(el => {
                const s = getComputedStyle(el);
                return s.borderLeftColor + '|' + s.borderLeftWidth;
              });
              probe.remove();
              return out;
            }"""
        )
        assert len(set(rails)) == 3, f"the three findings share a left rail: {rails}"
        assert all(r.split("|")[1] != "0px" for r in rails), "no rail is drawn at all"


class TestEveryTabRenders:
    """Cheap, and it catches the class of error that takes a whole view down."""

    @pytest.mark.parametrize(
        "name,view",
        [
            ("Process", "view-process"),
            ("Review queue", "view-queue"),
            ("Records", "view-records"),
            ("Insights", "view-insights"),
            ("Settings", "view-settings"),
        ],
    )
    def test_the_tab_shows_its_own_view_and_only_that(
        self, page: Page, name: str, view: str
    ) -> None:
        _tab(page, name)
        shown = page.evaluate(
            "() => [...document.querySelectorAll('.view')]"
            ".filter(v => getComputedStyle(v).display !== 'none').map(v => v.id)"
        )
        assert shown == [view], f"{name} showed {shown}"

    def test_no_console_errors_anywhere(self, page: Page) -> None:
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        for name in ("Process", "Review queue", "Records", "Insights", "Settings"):
            _tab(page, name)
        assert not errors, f"console errors: {errors}"


class TestTheBrowserIsToldNotToReuseStaleFiles:
    """The UI ships as three unversioned files with an ETag and no
    Cache-Control, so a browser reused them without asking. An edited
    stylesheet kept rendering the old one until a hard reload - which reads as
    "the fix did not work" rather than "the file is stale".
    """

    @pytest.mark.parametrize("path", ["/", "/app.js", "/styles.css"])
    def test_each_asset_must_be_revalidated(self, page: Page, live_server: str, path: str) -> None:
        response = page.request.get(live_server + path)
        assert response.status == 200
        cache_control = (response.headers.get("cache-control") or "").lower()
        assert "no-cache" in cache_control, (
            f"{path} came back with cache-control={cache_control!r}; a browser is "
            "free to reuse it without asking"
        )
