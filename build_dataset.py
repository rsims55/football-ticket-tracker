import pandas as pd
from schedule_fetcher import ScheduleFetcher
from rankings_fetcher import RankingsFetcher
from stadium_scraper import StadiumScraper
from rivalry_scraper import RivalryScraper
from ticket_pricer import TicketPricer
from datetime import datetime

class DatasetBuilder:
    def __init__(self, year=None, use_mock_tickets=True):
        self.year = year or datetime.now().year
        self.schedule = None
        self.use_mock_tickets = use_mock_tickets

    def build(self):
        # Step 1: Schedule
        schedule_fetcher = ScheduleFetcher(self.year)
        schedule = schedule_fetcher.fetch()

        # Step 2: Merge team rankings
        rankings_df = RankingsFetcher(self.year).fetch_and_load()
        if rankings_df is not None:
            rankings_df = rankings_df.rename(columns={"school": "rank_school", "rank": "rank"})

            schedule = schedule.merge(
                rankings_df[["rank_school", "rank"]],
                left_on="homeTeam",
                right_on="rank_school",
                how="left"
            ).rename(columns={"rank": "homeTeamRank"}).drop(columns=["rank_school"])

            schedule = schedule.merge(
                rankings_df[["rank_school", "rank"]],
                left_on="awayTeam",
                right_on="rank_school",
                how="left"
            ).rename(columns={"rank": "awayTeamRank"}).drop(columns=["rank_school"])
        else:
            schedule["homeTeamRank"] = None
            schedule["awayTeamRank"] = None
        
        # Step 2.5: Ranked matchup indicator
        schedule["isRankedMatchup"] = schedule["homeTeamRank"].notna() & schedule["awayTeamRank"].notna()

        # Step 3: Stadiums
        stadiums = StadiumScraper().scrape()
        schedule = schedule.merge(stadiums, left_on="homeTeam", right_on="school", how="left").drop(columns=["school"])

        # Step 4: Rivalries
        rivalries = RivalryScraper().scrape()
        schedule["isRivalry"] = schedule.apply(
            lambda row: row["awayTeam"] in rivalries.get(row["homeTeam"], []) or
                        row["homeTeam"] in rivalries.get(row["awayTeam"], []),
            axis=1
        )

        # Step 5: Ticket prices
        pricer = TicketPricer(use_mock=self.use_mock_tickets)
        ticket_data = []
        for _, row in schedule.iterrows():
            try:
                game_date = pd.to_datetime(row["startDateEastern"])
                team = row["homeTeam"]
                summary = pricer.get_summary(team, game_date)
            except Exception:
                summary = {}
            ticket_data.append(summary)

        ticket_df = pd.DataFrame(ticket_data)
        schedule = pd.concat([schedule.reset_index(drop=True), ticket_df.reset_index(drop=True)], axis=1)

        # Final: Keep only needed columns
        columns_to_keep = [
            "season", "week", "seasonType", "startDateEastern", "dayOfWeek", "kickoffTimeStr",
            "homeTeam", "awayTeam", "homeTeamRank", "awayTeamRank", "isRankedMatchup",
            "stadium", "capacity", "city", "state", "neutralSite",
            "isRivalry", "conferenceGame", "excitementIndex",
            "lowest_price", "average_price", "listing_count",
            "group_price_2", "group_price_4", "lowest_price_upper", "lowest_price_lower"
        ]

        schedule = schedule[columns_to_keep]

        self.schedule = schedule
        return schedule

    def save(self, filename=None):
        if self.schedule is None:
            return
        filename = filename or f"data/enriched_schedule_{self.year}.csv"
        self.schedule.to_csv(filename, index=False)
        print(f"✅ Enriched dataset saved to {filename}")

# ✅ Test run (silent except final print)
if __name__ == "__main__":
    builder = DatasetBuilder(use_mock_tickets=True)
    builder.build()
    builder.save()
