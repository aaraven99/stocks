def blend(prediction,mc):
 p=.55*prediction['probability']+.45*mc['p_plus_3']; return {**prediction,'probability':p,'expected_return':.5*prediction['expected_return']+.5*mc['expected_return']}
