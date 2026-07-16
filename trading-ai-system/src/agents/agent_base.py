"""Evidence-driven specialist agent base.

The agents are deliberately deterministic: they summarize quantitative evidence and
never create a trade signal or call an LLM.  The debate layer may use their structured
opinions for narrative/governance, while portfolio decisions continue to come from the
validated quantitative pipeline.
"""
from statistics import mean


class Agent:
    name = 'agent'
    stance = 'neutral'
    focus = 'general'

    def __init__(self, db=None):
        self.db = db

    @staticmethod
    def _clip(value, low=.05, high=.95):
        return float(max(low, min(high, float(value))))

    def _evidence_assess(self, context):
        candidates = list(context.get('candidate_evidence') or [])
        probabilities = [float(x.get('probability', .5)) for x in candidates]
        mc_returns = [float(x.get('mc_expected_return', 0.0)) for x in candidates]
        stress = [float(x.get('stress_penalty', 0.0)) for x in candidates]
        confidence = [float(x.get('data_confidence', x.get('confidence', .5))) for x in candidates]
        avg_probability = mean(probabilities) if probabilities else .5
        avg_mc_return = mean(mc_returns) if mc_returns else 0.0
        avg_stress = mean(stress) if stress else 0.0
        avg_data_confidence = mean(confidence) if confidence else .0
        risk = self._clip(context.get('risk', .5), 0.0, 1.0)
        regime = str(context.get('regime', 'sideways_chop'))
        bullish_regime = regime.startswith('bull')
        bearish_regime = regime.startswith('bear')

        # Directional evidence is transformed into confidence in this agent's stance.
        if self.stance == 'bull':
            score = .50 + .70 * (avg_probability - .50) + .35 * avg_mc_return
            score += .08 if bullish_regime else (-.08 if bearish_regime else 0.0)
            score -= .22 * risk + .35 * avg_stress
        elif self.stance == 'bear':
            score = .50 + .70 * (.50 - avg_probability) - .35 * avg_mc_return
            score += .08 if bearish_regime else (-.08 if bullish_regime else 0.0)
            score += .22 * risk + .35 * avg_stress
        else:
            score = .50 + .25 * abs(avg_probability - .50) + .08 * (1.0 - avg_data_confidence)
            score += .12 * risk if self.focus in {'risk', 'stress', 'compliance'} else 0.0
        score = self._clip(score)
        evidence = [
            {'source': 'candidate_predictions', 'value': round(avg_probability, 6), 'detail': f'{len(candidates)} candidates; mean P(positive)'},
            {'source': 'monte_carlo', 'value': round(avg_mc_return, 6), 'detail': 'mean expected return'},
            {'source': 'stress_testing', 'value': round(avg_stress, 6), 'detail': 'mean stress penalty'},
            {'source': 'data_quality', 'value': round(avg_data_confidence, 6), 'detail': 'mean data confidence'},
            {'source': 'regime', 'value': regime, 'detail': 'rules/HMM/cluster regime'},
            {'source': 'risk_governance', 'value': round(risk, 6), 'detail': 'current portfolio/model risk'},
        ]
        return {
            'agent': self.name,
            'stance': self.stance,
            'focus': self.focus,
            'score': score,
            'confidence': score,
            'evidence': evidence,
            'reasoning': f'{self.name} evaluated {self.focus} evidence: regime={regime}, '
                         f'mean_probability={avg_probability:.3f}, mean_mc_return={avg_mc_return:.3f}, '
                         f'mean_stress_penalty={avg_stress:.3f}, data_confidence={avg_data_confidence:.3f}.',
        }

    def dispatch(self, context):
        """Return the governed evidence assessment used by the debate engine."""
        return self._evidence_assess(context)

    def assess(self, context):
        # Kept as a public API for direct specialist use.  DebateEngine calls dispatch
        # so legacy specialist overrides cannot silently bypass evidence governance.
        return self._evidence_assess(context)

    def persist(self, date, payload):
        if self.db:
            self.db.upsert('agent_outputs', {'date': date, 'agent': self.name, 'payload_json': payload}, ['date', 'agent'])
