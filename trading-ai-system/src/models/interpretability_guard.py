def assess(explanation):
 dominant=max((abs(x['attribution']) for x in explanation),default=0); return {'passed':dominant<1000,'dominant_attribution':dominant}
