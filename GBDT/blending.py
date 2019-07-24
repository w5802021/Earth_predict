import pandas as pd
import numpy as np

I1 = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\gpI.csv')
I2 = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\GBDT\Xgboost_result/sub_xgb_no_shuffle.csv')
I3 = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\GBDT\Lightgbm_result/final_submission-light.csv')
I4 = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\GBDT\Lightgbm_result/sub_lgb_pear500_6f_shuffle_1.412.csv')
# I5 = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction/NN/nn_submission.csv')

submission = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction/input/sample_submission.csv')
submission['time_to_failure'] = (I1['time_to_failure'] + I2['time_to_failure'] + I3['time_to_failure'] + I4['time_to_failure'])/4

# for i in I1.index:
#     val = []
#     a = I1.ix[i, 'time_to_failure']
#     b = I2.ix[i, 'time_to_failure']
#     c = I3.ix[i, 'time_to_failure']
#     d = I5.ix[i, 'time_to_failure']
#     val.append(a)
#     val.append(b)
#     val.append(c)
#     val.append(d)
#     if max(val) > 10:
#         val.sort(reverse = True)
#         submission.ix[i,'time_to_failure'] = 0.4*val[0] + 0.4*val[1] + 0.2*val[2]
#     else:
#         submission.ix[i, 'time_to_failure'] = 0.2*a + 0.2*b + 0.2*c + 0.4*d

submission.to_csv('blending_xgb_ligh22_GPI_final.csv',index=False)
