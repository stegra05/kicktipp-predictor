#!/usr/bin/env python3
"""
Demo script to test the predictor components.
Run this to verify everything is working correctly.
"""

from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer


def main():
    print("="*60)
    print("3. LIGA PREDICTOR - DEMO")
    print("="*60)
    print("\nThis demo will test the data fetching capabilities.\n")

    # Test data fetcher
    print("1. Testing Data Fetcher...")
    print("-" * 60)

    data_fetcher = DataFetcher()

    # Get current season
    current_season = data_fetcher.get_current_season()
    print(f"✓ Current season: {current_season}/{current_season+1}")

    # Fetch current season matches
    print("\n2. Fetching current season matches...")
    print("-" * 60)

    matches = data_fetcher.fetch_season_matches(current_season)
    finished = [m for m in matches if m['is_finished']]
    upcoming = [m for m in matches if not m['is_finished']]

    print(f"✓ Total matches: {len(matches)}")
    print(f"✓ Finished matches: {len(finished)}")
    print(f"✓ Upcoming matches: {len(upcoming)}")

    # Show sample match
    if finished:
        print("\n3. Sample Finished Match:")
        print("-" * 60)
        sample = finished[-1]
        print(f"  {sample['home_team']} {sample['home_score']}:{sample['away_score']} {sample['away_team']}")
        print(f"  Date: {sample['date'].strftime('%Y-%m-%d %H:%M')}")
        print(f"  Matchday: {sample['matchday']}")

    # Show upcoming matches
    if upcoming:
        print("\n4. Next Upcoming Matches:")
        print("-" * 60)
        for match in upcoming[:3]:
            print(f"  {match['home_team']} vs {match['away_team']}")
            print(f"  Date: {match['date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"  Matchday: {match['matchday']}")
            print()

    # Test feature engineering
    if finished:
        print("5. Testing Feature Engineering...")
        print("-" * 60)

        feature_engineer = FeatureEngineer()

        # Create features for a few matches
        sample_matches = finished[-10:]
        features_df = feature_engineer.create_features_from_matches(sample_matches)

        print(f"✓ Created features for {len(features_df)} matches")
        print(f"✓ Number of features: {len(features_df.columns)}")
        print(f"✓ Sample features: {', '.join(list(features_df.columns)[:5])}...")

    # Get current table
    print("\n6. Current League Table (Top 5):")
    print("-" * 60)

    table = feature_engineer._calculate_table(matches)
    sorted_teams = sorted(table.items(), key=lambda x: x[1]['position'])

    print(f"{'Pos':<4} {'Team':<30} {'P':<4} {'W':<4} {'D':<4} {'L':<4} {'Pts':<4}")
    print("-" * 60)

    for i, (team, data) in enumerate(sorted_teams[:5], 1):
        print(f"{i:<4} {team:<30} {data['played']:<4} {data['won']:<4} "
              f"{data['drawn']:<4} {data['lost']:<4} {data['points']:<4}")

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nEverything is working correctly!")
    print("\nNext steps:")
    print("1. Run 'python train_model.py' to train the prediction models")
    print("2. Run 'python predict.py' to generate predictions")
    print("3. Run 'python src/web/app.py' to start the web interface")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. Internet connection is working")
        print("2. All dependencies are installed: pip install -r requirements.txt")
        print("3. OpenLigaDB API is accessible")
