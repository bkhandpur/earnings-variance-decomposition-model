# Synthetic offline sample

`synthetic_prices.csv` contains invented OHLCV observations on a weekday-only
calendar, January 2, 2023 through December 30, 2024. It is not a US exchange calendar.
No actual security or trade is represented. The fixture is distributed under this
repository's MIT license.

Columns: Date (YYYY-MM-DD session label), Open, High, Low, Close (positive price
units) and Volume (nonnegative units). Supply consistently split-adjusted OHLC,
including adjusted Close, for actual research; do not mix raw OHLC and adjusted
Close. Session labels are exchange-local dates, without a timezone or timestamp.
The CSV loader sorts dates, rejects duplicates and validates price ranges.

The synthetic close path is `100*exp(cumulative returns)`. For session index j,
returns are `.006*sin(1.7*j)+.004*cos(.37*j)`, except indices 85,148,211,274,337,400,463,
which receive .04,-.06,.025,-.08,.045,.03,-.055. Open is the preceding close; High
and Low bracket Open and Close by 0.4%; Volume is 1,000,000.

`synthetic_events.csv` has a `date` column with the first affected session of each
invented announcement. For real after-close announcements use the next session
as the effective date or `--announcement-offset 1`; before-open events use that
session. The default two-session window tolerates uncertain timing but includes
extra ordinary returns. Duplicate/nontrading announcement alignment is reported.
