# src/preparation/build_dataset.py
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

from fetchers.schedule_fetcher import ScheduleFetcher
from fetchers.rankings_fetcher import RankingsFetcher
from scrapers.stadium_scraper import StadiumScraper
from scrapers.rivalry_scraper import RivalryScraper

# Prefer TickPick
try:
    from scrapers.tickpick_pricer import TickPickPricer
    _HAS_TICKPICK = True
except Exception:
    _HAS_TICKPICK = False
    TickPickPricer = None  # type: ignore


class DatasetBuilder:
    """
    Build an enriched CFB schedule for the season:
    - schedule
    - rankings (home/away ranks + isRankedMatchup)
    - stadiums (capacity/city/state)
    - rivalry flag
    - (optional) current ticket prices via TickPick
    """
    def __init__(self, year: Optional[int] = None, use_mock_tickets: bool = True, include_prices: bool = False):
        self.year = year or datetime.now().year
        self.schedule: Optional[pd.DataFrame] = None
        self.use_mock_tickets = use_mock_tickets
        self.include_prices = include_prices

    def _merge_rankings(self, schedule: pd.DataFrame) -> pd.DataFrame:
        rankings_df = RankingsFetcher(self.year).fetch_and_load()
        if rankings_df is not None and not rankings_df.empty:
            rankings_df = rankings_df.rename(columns={"school": "rank_school", "rank": "rank"})
            # Home
            schedule = schedule.merge(
                rankings_df[["rank_school", "rank"]],
                left_on="homeTeam",
                right_on="rank_school",
                how="left"
            ).rename(columns={"rank": "homeTeamRank"}).drop(columns=["rank_school"])
            # Away
            schedule = schedule.merge(
                rankings_df[["rank_school", "rank"]],
                left_on="awayTeam",
                right_on="rank_school",
                how="left"
            ).rename(columns={"rank": "awayTeamRank"}).drop(columns=["rank_school"])
        else:
            schedule["homeTeamRank"] = pd.NA
            schedule["awayTeamRank"] = pd.NA

        schedule["isRankedMatchup"] = schedule["homeTeamRank"].notna() & schedule["awayTeamRank"].notna()
        return schedule

    def _merge_stadiums(self, schedule: pd.DataFrame) -> pd.DataFrame:
        stadiums = StadiumScraper().scrape()
        # Expecting: stadiums columns include ['school','stadium','capacity','city','state']
        schedule = schedule.merge(stadiums, left_on="homeTeam", right_on="school", how="left")
        if "school" in schedule.columns:
            schedule = schedule.drop(columns=["school"])
        return schedule

    def _apply_rivalries(self, schedule: pd.DataFrame) -> pd.DataFrame:
        rivalries: Dict[str, list] = RivalryScraper().scrape()
        schedule["isRivalry"] = schedule.apply(
            lambda row: row["awayTeam"] in rivalries.get(row["homeTeam"], []) or
                        row["homeTeam"] in rivalries.get(row["awayTeam"], []),
            axis=1
        )
        return schedule

    def _get_price_summary(self, pricer, row: pd.Series) -> Dict[str, Any]:
        """
        Tries multiple signatures to call TickPickPricer.get_summary.
        Expected keys: lowest_price, average_price, listing_count, source_url (optional)
        """
        event_id = row.get("event_id") or row.get("eventId")
        home = row.get("homeTeam")
        away = row.get("awayTeam")
        start_dt = pd.to_datetime(row.get("startDateEastern"), errors="coerce")

        try_order = [
            {"event_id": event_id, "home_team": home, "away_team": away, "start_dt": start_dt},
            {"event_id": event_id},
            {"home_team": home, "away_team": away, "start_dt": start_dt},
            {"home_team": home, "away_team": away},
        ]
        last_err = None
        for kwargs in try_order:
            try:
                clean = {k: v for k, v in kwargs.items() if v is not None}
                if not clean:
                    continue
                res = pricer.get_summary(**clean)
                if isinstance(res, dict) and res:
                    return res
            except Exception as e:
                last_err = e
                continue

        # If pricer fails, return NaNs so schema remains consistent
        return {
            "lowest_price": pd.NA,
            "average_price": pd.NA,
            "listing_count": pd.NA,
            "source_url": pd.NA,
            "_error": str(last_err) if last_err else pd.NA
        }

    def build(self) -> pd.DataFrame:
        # Step 1: Schedule
        schedule = ScheduleFetcher(self.year).fetch()

        # Normalize commonly used columns if needed
        # Expect at minimum: ['season','week','seasonType','startDateEastern','dayOfWeek','kickoffTimeStr','homeTeam','awayTeam',...]
        # Keep event_id if available
        if "eventId" in schedule.columns and "event_id" not in schedule.columns:
            schedule = schedule.rename(columns={"eventId": "event_id"})

        # Step 2: Merge rankings
        schedule = self._merge_rankings(schedule)

        # Step 3: Stadiums
        schedule = self._merge_stadiums(schedule)

        # Step 4: Rivalries
        schedule = self._apply_rivalries(schedule)

        # Step 5: Ticket prices (optional)
        if self.include_prices:
            if not _HAS_TICKPICK:
                raise ImportError(
                    "TickPickPricer not available. Ensure fetchers/tickpick_pricer.py exists "
                    "and is importable, or run with include_prices=False."
                )
            pricer = TickPickPricer(use_mock=self.use_mock_tickets)  # type: ignore
            ticket_rows = []
            for _, row in schedule.iterrows():
                ticket_rows.append(self._get_price_summary(pricer, row))
            ticket_df = pd.DataFrame(ticket_rows)
            schedule = pd.concat([schedule.reset_index(drop=True), ticket_df.reset_index(drop=True)], axis=1)
        else:
            # Ensure price columns exist even if not fetched here
            for col in ["lowest_price", "average_price", "listing_count", "source_url"]:
                if col not in schedule.columns:
                    schedule[col] = pd.NA

        # Provide placeholders for optional price-derivative columns if they don't exist
        for col in ["group_price_2", "group_price_4", "lowest_price_upper", "lowest_price_lower"]:
            if col not in schedule.columns:
                schedule[col] = pd.NA

        # Ensure excitementIndex present (if ScheduleFetcher didn’t include it)
        if "excitementIndex" not in schedule.columns:
            schedule["excitementIndex"] = pd.NA

        # Final: Keep only needed columns (plus event_id if present)
        columns_to_keep = [
            "season", "week", "seasonType", "startDateEastern", "dayOfWeek", "kickoffTimeStr",
            "homeTeam", "awayTeam", "homeTeamRank", "awayTeamRank", "isRankedMatchup",
            "stadium", "capacity", "city", "state", "neutralSite",
            "isRivalry", "conferenceGame", "excitementIndex",
            "lowest_price", "average_price", "listing_count",
            "group_price_2", "group_price_4", "lowest_price_upper", "lowest_price_lower"
        ]
        if "event_id" in schedule.columns:
            columns_to_keep = ["event_id"] + columns_to_keep

        missing = [c for c in columns_to_keep if c not in schedule.columns]
        for m in missing:
            schedule[m] = pd.NA

        schedule = schedule[columns_to_keep]

        self.schedule = schedule
        return schedule

    def save(self, filename: Optional[str] = None):
        if self.schedule is None:
            return
        filename = filename or f"data/enriched_schedule_{self.year}.csv"
        self.schedule.to_csv(filename, index=False)
        print(f"✅ Enriched dataset saved to {filename}")


if __name__ == "__main__":
    # Default: do NOT include prices here; use price logger for snapshots
    builder = DatasetBuilder(use_mock_tickets=True, include_prices=False)
    builder.build()
    builder.save()
