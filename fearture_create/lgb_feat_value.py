import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.svm import NuSVR, SVR
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from scipy.signal import hanning as han
from scipy.signal import hilbert, convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold,cross_val_score
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from GBDT import myhyperopt

# settings
warnings.filterwarnings('ignore')
np.random.seed(2019)
PATH = "E:/kaggle/kaggle-Lanl_Earthquake_Prediction/"


# logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

def load_feature_data():
    logger.info('Start read feature data')

    # train_X = pd.DataFrame()
    train_features = pd.read_csv(PATH + 'input/lanl-features/train_features.csv')
    train_features1 = pd.read_csv(PATH + 'fearture_create/train_features.csv')
    # train_X = pd.concat([train_features,train_features1],axis=1)
    train_X = train_features1.ix[:-1]

    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    # test_X = pd.DataFrame()
    test_features = pd.read_csv(PATH + 'input/lanl-features/test_features.csv')
    test_features1 = pd.read_csv(PATH + 'fearture_create/test_features.csv')
    # test_X = pd.concat([test_features,test_features1],axis=1)
    test_X = test_features1

    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    train_y = pd.read_csv(PATH + 'fearture_create/y_1419.csv')
    submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id')

    return scaled_train_X, scaled_test_X, train_y,submission

def run_model_lgbm(params,scaled_train_X, scaled_test_X, train_y,feature_col):
    logger.info('Run lgbm model')
    n_fold = 13
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    train_columns = feature_col

    oof = np.zeros(len(scaled_train_X))
    predictions = np.zeros(len(scaled_test_X))
    feature_importance_df = pd.DataFrame()

    # run model
    # a = folds.split(scaled_train_X, train_y.values)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print("fold {}".format(fold_))
        logger.info("fold {}".format(fold_))

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        clf = lgb.LGBMRegressor(**params, n_estimators = 200000, n_jobs=-1)

        clf.fit(X_tr, y_tr,eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',verbose=1000,early_stopping_rounds=1000)

        oof[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration_)
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = train_columns
        fold_importance_df["importance"] = clf.feature_importances_[:len(train_columns)]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # predictions
        predictions += clf.predict(scaled_test_X, num_iteration=clf.best_iteration_) / folds.n_splits
    # feature_importance_df.to_csv('feature_importance_denoise.csv')
    strLog = "CV score: {}".format(mean_absolute_error(train_y.values, oof))
    print(strLog)
    logger.info(strLog)

    return oof,predictions, feature_importance_df,mean_absolute_error(train_y.values, oof)

def plt_feature_importance(feature_importance_df):
    logger.info('Plot feature importance')
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:200].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 26))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

def submit(submission, predictions,i):
    # submission
    logger.info('Submisison')
    submission.time_to_failure = predictions
    submission.to_csv('./Lightgbm_result/submission_' + str(i) + '_hyperopt.csv', index=True)

def main(nrows=None):

    scaled_train_X, scaled_test_X, train_y, submission = load_feature_data()
    # feature_importance_df = pd.read_csv('feature_importance.csv')
    #
    # cols = list((feature_importance_df[["Feature", "importance"]]
    #                      .groupby("Feature")
    #                      .mean()
    #                      .sort_values(by="importance", ascending=False)[:].index))[:200]
    # cols.extend(scaled_train_X.columns[-20:])
    # dd = pd.read_excel(PATH + 'input/lanl-features/features_select2.xlsx',header=None)
    # dd.columns = ['feat']
    # cols = list(dd['feat']) + list(mfcccols)

    # best_feat = pd.read_csv('best_features.csv')
    # best_feat1 = best_feat[best_feat['importance'] >= 1]
    # cols1 = list(best_feat1['feat'])
    #
    # scaled_train_X = scaled_train_X[cols1]
    # scaled_test_X = scaled_test_X[cols1]

    # l = [0.5]
    # res = []
    # for i in l:
    #
    #     train_X_aug = data_augmentation(scaled_train_X,i)
    #     train_y_aug = data_augmentation(train_y,i)
    #     train_all = pd.concat([scaled_train_X, train_X_aug])
    #     y_all = pd.concat([train_y, train_y_aug])

    # lgbm_params = myhyperopt.quick_hyperopt(train_all, y_all, 'lgbm', 2500)

    lgbm_params = {'num_leaves': 128,
              'min_data_in_leaf': 79,
              'objective': 'gamma',
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": 0.8126672064208567,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1302650970728192,
              'reg_lambda': 0.3603427518866501,
              'feature_fraction': 0.1,
               'colsample_bytree': 1.0
              }

    oof,predictions, feature_importance,mae = run_model_lgbm(lgbm_params,scaled_train_X, scaled_test_X, train_y,scaled_train_X.columns)
    # run_model_xgb(scaled_train_X,scaled_test_X, train_y,scaled_train_X.columns)
    # submit(submission, predictions,'afterchose')
    # res.append(mae)
    # plt_feature_importance(feature_importance)
    # plt.figure(figsize=(16, 8))
    # plt.plot(train_y, color='g', label='y_train')
    # plt.plot(oof, color='r', label='lgb')
    # plt.legend()
    # plt.title('Predictions vs actual')
    # plt.show()
    # print(res)

if __name__ == "__main__":
    main()