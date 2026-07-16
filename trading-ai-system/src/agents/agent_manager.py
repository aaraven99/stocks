"""Registers every specialist in the mandated debate, while quantitative outputs remain decision inputs."""
from importlib import import_module
from agents.debate_engine import DebateEngine
SPECIALISTS=[
 ('market_regime_agent','MarketRegimeAgent'),('macro_agent','MacroAgent'),('technical_agent','TechnicalAgent'),('smc_agent','SMCAgent'),('sentiment_agent','SentimentAgent'),('montecarlo_agent','MonteCarloAgent'),('ml_agent','MLAgent'),('portfolio_agent','PortfolioAgent'),('stress_test_agent','StressTestAgent'),('risk_agent','RiskAgent'),('execution_agent','ExecutionAgent'),('compliance_agent','ComplianceAgent'),('llm_research_agent','LLMResearchAgent'),('final_decision_agent','FinalDecisionAgent'),('bull_team_agent','BullTeamAgent'),('bear_team_agent','BearTeamAgent'),('neutral_mediator_agent','NeutralMediatorAgent'),('cross_asset_agent','CrossAssetAgent'),('portfolio_risk_committee_agent','PortfolioRiskCommitteeAgent'),('event_risk_agent','EventRiskAgent'),('red_team_agent','RedTeamAgent'),('adversarial_agent','AdversarialAgent'),('causal_sanity_agent','CausalSanityAgent'),('research_lab_agent','ResearchLabAgent'),('signal_discovery_agent','SignalDiscoveryAgent'),('trade_lifecycle_agent','TradeLifecycleAgent')]
def run_debate(db,date,context):
 agents=[]
 for module,klass in SPECIALISTS:
  agent=getattr(import_module(f'agents.{module}'),klass)(db)
  agent.focus=module.replace('_agent','').replace('_',' ')
  agents.append(agent)
 return DebateEngine(db,agents).run(date,context)
