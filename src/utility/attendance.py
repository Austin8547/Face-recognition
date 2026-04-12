import datetime
from src.utility.database import DatabaseManager


_CAPTURE_WINDOWS: list[tuple[str, datetime.time, datetime.time]] = [
    ('h1', datetime.time(9,  40), datetime.time(9,  50)),
    ('h2', datetime.time(10, 40), datetime.time(10, 50)),
    ('h3', datetime.time(11, 55), datetime.time(12,  5)),
    ('h4', datetime.time(14, 10), datetime.time(14, 20)),
    ('h5', datetime.time(15, 10), datetime.time(15, 20)),
]


class AttendanceManager:

    def __init__(self):
        self.db = DatabaseManager()
        # {identifier -> hour_col} — who has been handled in the current window
        self._seen_this_hour: dict[str, str] = {}
        # Pre-populate today's attendance rows for ALL students (absent by default)
        inserted = self.db.initialize_daily_attendance()
        print(f"[Attendance] Today's session started — "
              f"{inserted} new student row(s) initialized (absent by default).")

    # ── Time-window helper ────────────────────────────────────────────────────

    def get_active_hour(self) -> str | None:
        """
        Return the active hour column ('h1'–'h5') if the current time falls
        inside any capture window, else return None.
        """
        now = datetime.datetime.now().time()
        for hour_col, win_start, win_end in _CAPTURE_WINDOWS:
            if win_start <= now <= win_end:
                return hour_col
        return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def update_attendance(self, identifier: str) -> float | None:
        """
        Called by inf_1.py every time a face is recognised.

        Args:
            identifier: The recognised name / roll_no from the embedding file.
                        (Will match roll_no once pkl files are renamed.)

        Returns:
            float  — today's fullday score (0.0 / 0.5 / 1.0) after any update.
            None   — if called outside all capture windows (frame is ignored).
        """

        # ── Step 1: Are we inside any capture window? ───────────────────────
        hour_col = self.get_active_hour()
        if hour_col is None:
            return None  # Outside all windows — nothing to record

        # ── Step 2: In-memory dedup — same person, same window ──────────────
        if self._seen_this_hour.get(identifier) == hour_col:
            return None  # Already processed this student for this hour

        # ── Step 3: Guarantee a row exists for today ─────────────────────────
        self.db.ensure_today_row(identifier)

        # ── Step 4: Idempotency — hour already in DB (e.g. after app restart)─
        if self.db.is_hour_marked(identifier, hour_col):
            self._seen_this_hour[identifier] = hour_col
            return self.db.get_daily_score(identifier)

        # ── Step 5: Write to DB — mark hour + recalc section flags ───────────
        self.db.mark_hour(identifier, hour_col)
        print(f"[Attendance] ✓ {identifier} → {hour_col} marked "
              f"({datetime.datetime.now().strftime('%H:%M:%S')})")

        # ── Step 6: Update in-memory dedup ───────────────────────────────────
        self._seen_this_hour[identifier] = hour_col

        # ── Step 7: Return fresh score to caller (displayed on-screen) ───────
        return self.db.get_daily_score(identifier)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Graceful shutdown — closes the DB connection."""
        self.db.close()
