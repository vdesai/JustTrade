#!/bin/bash
# Daily paper trader runner. Schedule via launchd or cron.
cd "$(dirname "$0")"
source .venv/bin/activate
python -m event_bot.paper_trader >> data/paper_trader.log 2>&1
