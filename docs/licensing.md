# Licensing and reuse policy

The project is MIT-licensed. It uses dependencies through normal package installation, rather
than copying code from reviewed repositories. This intentionally avoids the copyleft obligations
of attractive but unsuitable candidates such as Backtrader (GPL-3.0), Freqtrade (GPL-3.0),
Backtesting.py (AGPL-3.0), Lumibot (GPL-3.0), and OpenAlgo (AGPL-3.0).

Every new dependency must be recorded in `THIRD_PARTY_NOTICES.md`, checked for license
compatibility, and isolated behind an adapter where it interacts with market data or execution.
Data access permissions are separate from software licenses: a library does not grant rights to
redistribute a vendor's market data.

