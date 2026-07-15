def build(report):
 lines=[f"# Systematic Swing Research — {report['date']}",f"Regime: **{report['regime']}** | Decision: **{report['decision']}**",'', '|Ticker|Score|P(positive)|MC EV|Weight|','|-|-|-|-|-|']
 lines += [f"|{x['ticker']}|{x['score']:.3f}|{x['prediction']['probability']:.1%}|{x['mc']['expected_return']:.2%}|{x['weight']:.1%}|" for x in report.get('picks',[])]
 lines += ['', '> Research watchlist only; estimates are probabilistic and do not promise returns.']; return '\n'.join(lines)
