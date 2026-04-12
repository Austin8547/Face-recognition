import psycopg2



_MRG_HOURS = ['h1', 'h2', 'h3']    # Morning section
_EVG_HOURS = ['h4', 'h5']           # Evening section
_ALL_HOURS  = _MRG_HOURS + _EVG_HOURS


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection to attendence_db."""
    return psycopg2.connect(
        dbname   = "attendence_db",
        user     = "user_id",
        password = "your_password",
        host     = "localhost",
        port     = "5432",
    )


# ── DatabaseManager ───────────────────────────────────────────────────────────

class DatabaseManager:

    def __init__(self):
        self.conn = get_connection()
        self.conn.autocommit = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_mark_sql(self, hour_col: str) -> str:
        """
        Build the UPDATE SQL for marking one hour as attended.

        PostgreSQL evaluates all SET expressions against the *original* row
        values before any change is applied in the same statement.  If we
        naively wrote:
            SET h1 = TRUE, mrg_section = (h1 AND h2 AND h3)
        the mrg_section expression would use the OLD value of h1 (FALSE).

        Fix: substitute the target column with the literal TRUE in every
        derived expression so the calculation uses the new value.
        """
        if hour_col not in _ALL_HOURS:
            raise ValueError(
                f"Invalid hour column: '{hour_col}'. Must be one of {_ALL_HOURS}"
            )

        # Replace the target hour with TRUE in each section's expression
        mrg_parts = ['TRUE' if h == hour_col else h for h in _MRG_HOURS]
        evg_parts = ['TRUE' if h == hour_col else h for h in _EVG_HOURS]

        mrg_expr = ' AND '.join(mrg_parts)   # e.g. "TRUE AND h2 AND h3"
        evg_expr = ' AND '.join(evg_parts)   # e.g. "h4 AND h5"

        return f"""
            UPDATE attendance
            SET
                {hour_col}   = TRUE,
                mrg_section  = ({mrg_expr}),
                eveg_section = ({evg_expr}),
                fullday      = CASE
                                 WHEN ({mrg_expr}) AND ({evg_expr}) THEN 1.0
                                 WHEN ({mrg_expr}) OR  ({evg_expr}) THEN 0.5
                                 ELSE 0.0
                               END
            WHERE roll_no = %s AND attendance_date = CURRENT_DATE
        """

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize_daily_attendance(self) -> int:
        """
        Ensure EVERY student in the `students` table has an attendance row
        for today.  New rows default to all hours FALSE and fullday=0.0
        (i.e. absent).  Already-existing rows are left untouched.

        Returns the number of new rows inserted.  Call once at app startup.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance (roll_no, attendance_date)
                SELECT roll_no, CURRENT_DATE FROM students
                ON CONFLICT (roll_no, attendance_date) DO NOTHING
                """
            )
            inserted = cur.rowcount
        self.conn.commit()
        return inserted

    def ensure_today_row(self, roll_no: str) -> None:
        """
        Insert a blank attendance row for today for a single student.
        Safe to call multiple times — ON CONFLICT DO NOTHING.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance (roll_no, attendance_date)
                VALUES (%s, CURRENT_DATE)
                ON CONFLICT (roll_no, attendance_date) DO NOTHING
                """,
                (roll_no,),
            )
        self.conn.commit()

    def is_hour_marked(self, roll_no: str, hour_col: str) -> bool:
        """
        Return True if the given hour column is already TRUE for today's row.
        Returns False if the row doesn't exist yet or the column is NULL/FALSE.
        """
        if hour_col not in _ALL_HOURS:
            raise ValueError(f"Invalid hour column: '{hour_col}'")

        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {hour_col} FROM attendance "
                f"WHERE roll_no = %s AND attendance_date = CURRENT_DATE",
                (roll_no,),
            )
            row = cur.fetchone()

        return bool(row and row[0])

    def mark_hour(self, roll_no: str, hour_col: str) -> None:
        """
        Mark the given hour as attended and atomically recalculate
        mrg_section, eveg_section, and fullday for today's row.

        Assumes ensure_today_row() has already been called.
        """
        sql = self._build_mark_sql(hour_col)
        with self.conn.cursor() as cur:
            cur.execute(sql, (roll_no,))
        self.conn.commit()

    def get_daily_score(self, roll_no: str) -> float:
        """
        Return today's fullday score (0.0, 0.5, or 1.0) for the student.
        Returns 0.0 if no row exists yet.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT fullday FROM attendance "
                "WHERE roll_no = %s AND attendance_date = CURRENT_DATE",
                (roll_no,),
            )
            row = cur.fetchone()

        if row is None or row[0] is None:
            return 0.0
        return float(row[0])

    def close(self) -> None:
        """Close the database connection gracefully."""
        if self.conn and not self.conn.closed:
            self.conn.close()
