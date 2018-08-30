"""
Data Science utilities: bbalaji8@gmail.com; github.com/bdanalytics
"""

## Import packages
import os
os.system('clear')
# python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'

import sys
print('Running %s...' % (os.path.basename(sys.argv[0])))
print("  Python version:", sys.version)
print("  sys.path:", sys.path)
# sys.path.append('/Library/Python/2.7/site-packages')
if not os.path.isdir('assets'):
#     print("assets dir does not exist")
    os.makedirs('assets')

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)
print("\nStep 0: Import packages")
# print("  Importing myDSUtl...")
import numpy as np;  print("  numpy  version:", np.__version__ )
import pandas as pd; print("  pandas version:", pd.__version__ )
import pprint as pp; #print("  pandas version:", pd.__version__ )
import random;       #print("  random version:", random.__version__ )
import time as tm;   #print("  random version:", random.__version__ )

## Foundation utilities

def myDSUtlFormatNumber(x):
    if (type(x) is not type(np.float(0.0))):
        xCnv = np.float(x)
    else:
        xCnv = x
    assert xCnv is not np.nan, \
        f"myDSUtlFormatNumber: x: %s| cannot be converted to a number"%(x)    

    return f"%0.2e"%(xCnv)

# print('myDSUtlFormatNumber:%s'%([(x, myDSUtlFormatNumber(x)) for x in 
#                                  [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]]))
# print('myDSUtlFormatNumber:%s'%([(x, myDSUtlFormatNumber(x)) for x in 
#                                  ['1000', 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]]))
# print('myDSUtlFormatNumber:%s'%([(x, myDSUtlFormatNumber(x)) for x in 
#                                  ['1x', 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]]))

def myDSUtlGetPrimes(nLen = 100):
    """
    Copied from https://howchoo.com/g/ztk0mzq0mdy/generate-a-list-of-primes-numbers-in-python
    """
    n = int(nLen) ** 2
    assert (n <= 1e8), f'n:%0.4e too large'%(n)

    noprimes = set(j for i in range(2, 8) for j in range(i * 2, n, i))
    primes = [x for x in range(2, n) if x not in noprimes]
    assert len(primes) >= nLen, f"myDSUtlGetPrimes: len(primes):%d < nLen: %d" % (len(primes), nLen)
    return primes[:int(nLen)]

glbRndSeeds = {'primes': myDSUtlGetPrimes(), 'ix' : 2}

def myDSUtlGetNextRandomSeed(verbose = False):
    assert glbRndSeeds['ix'] < len(glbRndSeeds['primes']), \
        f"myDSUtlGetNextRandomSeed: glbRndSeeds['ix']:%d >= len(glbRndSeeds['primes']):%d" % (
            glbRndSeeds['ix'], len(glbRndSeeds['primes']))

    # random.seed(glbRndSeeds['primes'][glbRndSeeds['ix']])
    seed = glbRndSeeds['primes'][glbRndSeeds['ix']]
    glbRndSeeds['ix'] += 1

    if verbose:
        print('myDSUtlGetNextRandomSeed: seed:%d' % (seed))
    return seed

## Pandas utilities

# def myDSUtlDisplayDf(df, indent = 2):
#     print("\n%sDf: %s: shape: %s" % (' ' * indent, df.myId, df.shape))
#     print(df.describe(include = 'all'))
#     print("%s  sample records:" % (' ' * indent, ))
#     print(df.head(10))
#     print(df.tail(10))
def myDSUtlDisplayDf(df, label = None, indent = 2, nRows = 10):
    if (label is None) and (hasattr(df, "myId")):
        label = df.myId
    print("\n%sDf: %s: shape: %s" % (' ' * indent, label, df.shape))
    print(df.describe(include = 'all'))
    print("%s  sample records:" % (' ' * indent, ))
    print(df.head(nRows))
    if (df.shape[0] > nRows):
        print(df.tail(nRows))
        
    # Displays non-valid fill rates for columns    
    df.isnull().sum()/len(df)    

# ###	Process
#
# ###	01.	load data
# def myint_with_commas(x):
#
#     import numpy as np
#
#     if type(x) not in [type(0), type(0L), type(np.int64(0))]:
#         raise TypeError("Parameter must be an integer; instead type({0})={1}".format(x, type(x)))
#     if x < 0:
#         return '-' + myint_with_commas(-x)
#     result = ''
#     while x >= 1000:
#         x, r = divmod(x, 1000)
#         result = ",%03d%s" % (r, result)
#     return "%d%s" % (x, result)
#
# def mypred_var_count_print(pandas_obj, pred_varname):
#
#     import pandas as pd
#
#     if hasattr(pandas_obj, "name"):
#         print "			{0} in {1}:".format(pred_varname, pandas_obj.name)
#     else:
#         print "			{0} in unnamed object:".format(pred_varname)
#
#     if isinstance(pandas_obj, pd.SparseDataFrame):
#         pandas_obj = pandas_obj[pred_varname].to_dense()
#
#     if isinstance(pandas_obj, pd.DataFrame):
#         pred_var_count = pd.DataFrame(pandas_obj[pred_varname].value_counts(), columns=['count'])
#     elif isinstance(pandas_obj, pd.Series):
#         pred_var_count = pd.DataFrame(pandas_obj.value_counts(), columns=['count'])
#     else:
#         print "	mypred_var_count_print: expecting a pandas object, recd:{0}".format(pandas_obj)
#         exit(TypeError)
#
#     pred_var_count['counts %'] = (pred_var_count['count'] / float(pred_var_count['count'].sum())) * 100
#     #pp.pprint(pred_var_count)
#     #print "{0}".format(pred_var_count.to_string(float_format=lambda x: '%0.4f' % x))
#     print "{0}".format(pred_var_count.to_string(formatters=[myint_with_commas, lambda x: "%0.2f" % x]))
#
# def myimport_data(csv_filename, entity_name, pred_varname, index_col=None):
#
#     import pandas as pd
#
#     try:
#         entity = pd.read_csv(csv_filename, index_col=index_col)
#     except IOError:
#         print "		file {0} does not exist - exiting script".format(csv_filename)
#         exit(IOError)
#
#     print "		read {0}:({1:,},{2:,})".format(csv_filename, entity.shape[0], entity.shape[1])
#     entity.name = entity_name
#     if pred_varname in entity.columns:
#         mypred_var_count_print(entity, pred_varname)
#     else:
#         print "			prediction variable:{0} not in {1}".format(pred_varname, csv_filename)
#
#     return entity
#
# ###	02.	clean data
# ### 02.1	inspect data
# ### 02.2	fill missing data
# ###	02.3	drop cols that still contain NaNs
#
# def mydrop_na(data_frame, pred_varname):
#
#     new_df = data_frame.dropna(axis=1)
#     new_df.name = data_frame.name
#     print "		After dropping NaNs in {0}:".format(new_df.name)
#     mypred_var_count_print(new_df, pred_varname)
#
#     return new_df
#
# ###	03.	extract features
# ### 03.1	convert non-numerical features to numeric features
# ### 03.2 	create feature combinations
# ###	04. transform features
# ### 04.1	collect all numeric features
# ###	04.2	remove row keys & prediction variable
# ### 04.3	remove features that should not be part of estimation
# ###	04.4	remove features / create feature combinations for highly correlated features
#
# def myremove_corr_feats(entity, features, pred_varname, random_varname):
#
#     import copy
#     import numpy as np
#     import pandas as pd
#     from statsmodels.stats import outliers_influence
#     import pprint
#     pp = pprint.PrettyPrinter(indent=4)
#
#
#     uncorr_features = copy.copy(features)
#     print "		before collinearity cleanup:condition number of features matrix = {0:,}".format(int(np.linalg.cond(entity[uncorr_features])))
#
#     uncorr_features_vifs = []
#     for pos, feat in enumerate(uncorr_features):
#         uncorr_features_vifs.append(outliers_influence.variance_inflation_factor(entity[uncorr_features].values, pos))
#
#     #	VIF > 10 indicates serious collinearity
#     uncorr_features_series = pd.Series(uncorr_features_vifs, index=uncorr_features)
#     uncorr_features_series = uncorr_features_series.order(ascending=False)
#     print "		vifs:"
#     #pp.pprint(uncorr_features_series)
#     print "{0}".format(uncorr_features_series.to_string(float_format=lambda x: "%0.3f" % x))
#
#     #chk_features = uncorr_features_series[uncorr_features_series >= 10].index.tolist()
#     chk_features = copy.copy(uncorr_features)
#     if random_varname not in set(chk_features):
#         chk_features.append(random_varname)
#
#     chk_features.append(pred_varname)
#
#     features_corr = entity[chk_features].corr()
#     #print "						features correlation:"
#     #pp.pprint(feat_corr)
#
#     remove_features = set([])
#     for feat_i in range(features_corr.shape[0]):
#         if features_corr.columns[feat_i] in remove_features:
#                 continue
#         features_corr_row = features_corr.ix[feat_i]
#         for feat_j in range(features_corr.shape[0]):
#             if features_corr.columns[feat_j] in remove_features:
#                 continue
#             if feat_i <= feat_j:
#                 continue
#             if abs(features_corr_row[feat_j]) >= 0.6:
#                 print "\n							corr({0},{1}) = {2:0.4f}".format(
#                     features_corr.index[feat_i], features_corr.columns[feat_j], features_corr.ix[feat_i, feat_j])
#                 print "									corr({0},{1}) = {2:0.4f}".format(
#                     pred_varname, features_corr.columns[feat_i], features_corr.ix[pred_varname, feat_i])
#                 print "									corr({0},{1}) = {2:0.4f}".format(
#                     pred_varname, features_corr.columns[feat_j], features_corr.ix[pred_varname, feat_j])
#                 if abs(features_corr.ix[pred_varname, feat_i]) > abs(features_corr.ix[pred_varname, feat_j]):
#                     remove_features |= set([features_corr.columns[feat_j]])
#                 else:
#                     remove_features |= set([features_corr.columns[feat_i]])
#
#     remove_features.discard(random_varname)
#     print "						removing features:"
#     pp.pprint(remove_features)
#
#     for feat in remove_features:
#         if feat in set([]):	# Override feature removal
#             continue
#
#         uncorr_features.remove(feat)
#         chk_features.remove(feat)
#
#     features = uncorr_features
#     print "		uncorrelated feats:"
#     pp.pprint(features)
#     # check correlations & remove features until all corrs less than threshold ?
#     print "		 after collinearity cleanup:condition number of features matrix = {0:,}".format(int(np.linalg.cond(entity[features])))
#
#     return features
#
# ### 04.5	scale / normalize selected features for data distribution requirements in various models
# ###	05.	build training and test data
# ### 05.1	simple shuffle sample
# """ refactor as function
# features.append(rowkey_varname)
# tmp_train_X, tmp_test_X, train_y, test_y = cross_validation.train_test_split(entity[features].values,
#                                                                              entity[pred_varname].values,
#                                                                              test_size=.3)
#
# datadict_test = {}
# for feat in range(len(features)):
#     datadict_test[features[feat]] = tmp_test_X[:, feat]
#
# entity_test = pd.DataFrame(datadict_test)
# test_X = np.delete(tmp_test_X, len(features) - 1, 1)
# train_X = np.delete(tmp_train_X, len(features) - 1, 1)
# features.remove(rowkey_varname)
# pred_X = predict[features].values
# """
#
# ### 05.2	stratified shuffle sample
#
# def mybuild_stratified_samples(entity, pred_varname):
#
#     # sample entity into train, validate, train_plus_validate, test data frames
#     #	assumptions:
#     #		pred_varname is a binomial classification with values {0, 1}
#     #		model is MultivariateGaussian
#     #	returns:
#     #		train_entity:				60% of randomized sample of pred_varname = 0
#     #									10% of randomized sample of pred_varname = 1
#     #
#     #		validate_entity:			20% of randomized sample of pred_varname = 0
#     #									45% of randomized sample of pred_varname = 1
#     #
#     #		train_plus_validate_entity:	80% of randomized sample of pred_varname = 0
#     #									55% of randomized sample of pred_varname = 1
#     #
#     #		test_entity:				20% of randomized sample of pred_varname = 0
#     #									45% of randomized sample of pred_varname = 1
#
#     import random
#     import pandas as pd
#
#     class0_entity = entity[entity[pred_varname] == 0]
#     class1_entity = entity[entity[pred_varname] == 1]
#
#     train_rows = random.sample(class0_entity.index, int(class0_entity.shape[0] * 0.6))
#     train_class0_entity = class0_entity.ix[train_rows]
#     non_train_class0_entity = class0_entity.drop(train_rows)
#     validate_rows = random.sample(non_train_class0_entity.index, int(non_train_class0_entity.shape[0] * 0.5))
#     validate_class0_entity = non_train_class0_entity.ix[validate_rows]
#     test_class0_entity = non_train_class0_entity.drop(validate_rows)
#
#     train_rows = random.sample(class1_entity.index, int(class1_entity.shape[0] * 0.1))
#     train_class1_entity = class1_entity.ix[train_rows]
#     non_train_class1_entity = class1_entity.drop(train_rows)
#     validate_rows = random.sample(non_train_class1_entity.index, int(non_train_class1_entity.shape[0] * 0.5))
#     validate_class1_entity = non_train_class1_entity.ix[validate_rows]
#     test_class1_entity = non_train_class1_entity.drop(validate_rows)
#
#     train_entity = pd.concat([train_class0_entity, train_class1_entity])
#     validate_entity = pd.concat([validate_class0_entity, validate_class1_entity])
#     test_entity = pd.concat([test_class0_entity, test_class1_entity])
#     train_plus_validate_entity = pd.concat([train_entity, validate_entity])
#
#     train_entity.name = "train_" + entity.name
#     mypred_var_count_print(train_entity, pred_varname)
#     validate_entity.name = "validate_" + entity.name
#     mypred_var_count_print(validate_entity, pred_varname)
#     train_plus_validate_entity.name = "train+validate_" + entity.name
#     mypred_var_count_print(train_plus_validate_entity, pred_varname)
#     test_entity.name = "test_" + entity.name
#     mypred_var_count_print(test_entity, pred_varname)
#
#     return train_entity, validate_entity, train_plus_validate_entity, test_entity
#
# ###	05.3	cross-validation sample
# ### 06.	select models
# ### 06.1	select base models
# ###	06.1.1		regression models
# ###	06.1.2		classification models
#
# class myMultivariateGaussianClassifier:
#     # Theory in Andrew Ng's Coursera class
#     # 	ensure fitting does not include validation observations
#
#     def __init__(self):
#
#         self.epsilon = 0.0
#         self.feature_importances_ = None
#         self.mu = None
#         self.sq_sigma = None
#
#     def _get_Gaussian_proba_estimate(self, X):
#
#         import numpy as np
#
#         n_features = X.shape[1]
#         P_X = np.zeros([X.shape[0], n_features])
#         for feat_ix in range(n_features):
#             P_X[:,feat_ix] = ((1.0 / (((2 * np.pi) ** 0.5) * (self.sq_sigma[feat_ix] ** 0.5))) *
#                                 np.exp(-((X[:, feat_ix] - self.mu[feat_ix]) ** 2) / (2 * self.sq_sigma[feat_ix])))
#         #print "myMultivariateGaussianClassifier._get_Gaussian_proba_estimate: P_X[:5,:] = "
#         #pp.pprint(P_X[:5])
#         p_X = np.cumprod(P_X, axis=1)[:, n_features-1]
#         #print "myMultivariateGaussianClassifier._get_Gaussian_proba_estimate: p_X[:5,:] = "
#         #pp.pprint(p_X[:5])
#         return p_X
#
#     def fit(self, X, y):
#
#         import numpy as np
#         from sklearn import metrics
#
#         n_features = X.shape[1]
#         self.feature_importances_ = np.zeros(n_features)
#         #print "myMultivariateGaussianClassifier.fit: n_samples = {0:,}; n_features = {1:,}".format(n_samples, n_features)
#         y_eq_zero_X = X[y == 0]
#         y_eq_ones_X = X[y == 1]
#         #print "myMultivariateGaussianClassifier.fit: y_eq_zero_n_samples = {0:,}; y_eq_ones_n_samples = {1:,}".format(y_eq_zero_X.shape[0], y_eq_ones_X.shape[0])
#         self.mu = np.mean(y_eq_zero_X, axis=0)
#         #print "myMultivariateGaussianClassifier.fit: mu = "
#         #pp.pprint(self.mu)
#         self.sq_sigma = np.var(y_eq_zero_X, axis=0)
#         #print "myMultivariateGaussianClassifier.fit: sq_sigma = "
#         #pp.pprint(self.sq_sigma)
#         y_eq_zero_p_X = self._get_Gaussian_proba_estimate(y_eq_zero_X)
#         y_eq_ones_p_X = self._get_Gaussian_proba_estimate(y_eq_ones_X)
#         p_X = self._get_Gaussian_proba_estimate(X)
#         dist_p_X = np.zeros([len(range(10, 110, 10)), 6])
#         for pos, percentile in enumerate(range(10, 110, 10)):
#             dist_p_X[pos, 0] = percentile
#             dist_p_X[pos, 1] = np.percentile(p_X, percentile)
#             dist_p_X[pos, 2] = np.percentile(y_eq_zero_p_X, percentile)
#             dist_p_X[pos, 3] = np.percentile(y_eq_ones_p_X, percentile)
#             predict_y = p_X < dist_p_X[pos, 3]
#             dist_p_X[pos, 4] = sum(predict_y)	# len(predict_y == True) does not work
#             dist_p_X[pos, 5] = metrics.f1_score(y, predict_y)
#
#         #print "myMultivariateGaussianClassifier.fit: dist_p_X = "
#         #pp.pprint(dist_p_X)
#
#         #print "myMultivariateGaussianClassifier.fit: f1_score: min = {0:0.4f}; max = {1:0.4f}".format(np.min(dist_p_X[:,5]), np.max(dist_p_X[:,5]))
#         epsilon_pos = np.argmax(dist_p_X[:,5])
#         self.epsilon = dist_p_X[epsilon_pos, 3]
#         #print "myMultivariateGaussianClassifier.fit: percentile = {0}; epsilon = {1}".format(dist_p_X[epsilon_pos, 0], self.epsilon)
#         return self
#
#     def score(self, X, y):
#
#         from sklearn import metrics
#
#         predict_y = self._get_Gaussian_proba_estimate(X) < self.epsilon
#         score = metrics.accuracy_score(y, predict_y)
#         return score
#
#     def predict(self, X):
#         p_X = self._get_Gaussian_proba_estimate(X)
#         predict_y = p_X < self.epsilon
#         return predict_y
#
# class myKNeighborsClassifier():
#     # Scale features & predict utilizing unscaled data fed to KNeighborsClassifier
#
#     def __init__(self, **kwargs):
#
#         from sklearn import neighbors
#
#         self.base_model = neighbors.KNeighborsClassifier(**kwargs)
#         self.features_min = None
#         self.features_max = None
#
#     def _scale(self, X):
#
#         import numpy as np
#
#         #scaled_X = X.copy()
#         #for col in range(X.shape[1]):
#         #	scaled_X[:,col] = ((X[:,col] * 1.0) - self.features_min[col]) / \
#         #					  (self.features_max[col] - self.features_min[col])
#
#         return ((X * 1.0) - self.features_min) / (self.features_max - self.features_min)
#
#     def fit(self, X, y):
#
#         import numpy as np
#
#         self.features_min = np.min(X, axis=0)
#         #print "myKNeighborsClassifier.fit: features_min = "
#         #pp.pprint(self.features_min)
#         self.features_max = np.max(X, axis=0)
#         #print "myKNeighborsClassifier.fit: features_max = "
#         #pp.pprint(self.features_max)
#         #print "myKNeighborsClassifier.fit: X[-5:] = "
#         #pp.pprint(X[-5:])
#         scaled_X = self._scale(X)
#         #print "myKNeighborsClassifier.fit: scaled_X[-5:] = "
#         #pp.pprint(scaled_X[-5:])
#         return self.base_model.fit(scaled_X, y)
#
#     def score(self, X, y):
#         return self.base_model.score(self._scale(X), y)
#
#     def predict(self, X):
#         return self.base_model.predict(self._scale(X))
#
# ###	06.1.3		clustering models
# ###	06.1.4		dimensionality reduction models
# ### 06.2	select ensemble models
# ###	07.	design models
# ### 07.1	select significant features
# def myselect_significant_features(entity, features, pred_varname, random_varname):
#
#     from sklearn import feature_selection
#     import pandas as pd
#     import pprint
#     pp = pprint.PrettyPrinter(indent=4)
#     import copy
#
#     if isinstance(entity, pd.SparseDataFrame):
#         feat_f_scores, feat_p_vals = feature_selection.f_classif(entity[features].to_dense().values
#                                                                 ,entity[pred_varname].to_dense().values)
#     else:
#         feat_f_scores, feat_p_vals = feature_selection.f_classif(entity[features].values
#                                                                 ,entity[pred_varname].values)
#
#     features_series = pd.Series(data=feat_p_vals, index=features)
#     features_series = features_series.order()
#     print "		feature p-values:"
#     #pp.pprint(features_series)
#     print "{0}".format(features_series.to_string(float_format=lambda x: '%0.4f' % x))
#
#     features = list(features_series[features_series <= 0.05].index)
#     return features
#
# def myplot_significant_features(entity, features, pred_varname, random_varname, show=False, exp_prefix=None):
#     import myplots
#     import matplotlib.pyplot as plt
#     import copy
#     import pandas as pd
#
# #	return features
# #""" Remove comments tokens for plot creation
#     scatter_plot_columns = copy.copy(features)
#     scatter_plot_columns.append(random_varname)
#     if isinstance(entity, pd.SparseDataFrame):
#         scatter_plot_entity = entity[scatter_plot_columns].to_dense()
#     else:
#         scatter_plot_entity = entity[scatter_plot_columns]
#
#     #scatter_plot_entity = entity[list(entity.describe().columns)]
#     #for col in scatter_plot_entity.columns:
#     #	if col in [random_varname] or col in scatter_plot_columns:
#     #		continue
#     #	else:
#     #		#print "dropping col:{0}".format(col)
#     #		# del scatter_plot_entity[col] seems to corrupt the df; get_numeric_data() crashes
#     #		scatter_plot_entity = scatter_plot_entity.drop(col, 1)
#
#     # this crashes in myscatter_matrix if removed from here
#     df = scatter_plot_entity._get_numeric_data()
#     import pandas.core.common as com
#     mask = com.notnull(df)
#
#     #pd.tools.plotting.scatter_matrix(scatter_plot_entity, alpha=0.2, diagonal='hist')
#     if isinstance(entity, pd.SparseDataFrame):
#         myplots.myscatter_matrix(scatter_plot_entity, pred_values=entity[pred_varname].to_dense(), alpha=0.2, diagonal='hist')
#     else:
#         myplots.myscatter_matrix(scatter_plot_entity, pred_values=entity[pred_varname], alpha=0.2, diagonal='hist')
#     if show:
#         plt.show()
#
#     if exp_prefix is not None:
#         exp_filename = exp_prefix + "scatter_entity" + '.png'
#         print "		exporting plot:{0} ...".format(exp_filename)
#         plt.savefig(exp_filename, dpi=200)
#
#     return features
# #"""
#
# ###	07.1.1		add back in key features even though they might have been eliminated
# ###	07.2	identify model parameters (e.g. # of neighbors for knn, # of estimators for ensemble models)
# ###	08.	run models
# ###	08.1	fit on simple shuffled sample
# """ refactor as function
# models_cols_floats = 	['score'			# model
#                         ,'mislabels'		# key metric
#                                             # metrics module
#                         ,'f1_score', 'roc_auc_score', 'zero_one_loss'
#                         ,'accuracy_score', 'average_precision_score', 'precision_score', 'recall_score'
#                         ]
# for i in range(len(models_cols_floats)):
#     models[models_cols_floats[i]] = float(0.0)
#
# models['feature_importances'] = ""
#
# train_X = train_entity[features].values
# train_y = train_entity[pred_varname].values
# validate_X = validate_entity[features].values
# validate_y = validate_entity[pred_varname].values
# train_plus_validate_X = train_plus_validate_entity[features].values
# train_plus_validate_y = train_plus_validate_entity[pred_varname].values
# test_X = test_entity[features].values
# test_y = test_entity[pred_varname].values
# for model_ix, model_row in models.iterrows():
#     model = model_row['model']
#     model.fit(train_plus_validate_X, train_plus_validate_y)
#     models.ix[model_ix, 'score'] = model.score(train_plus_validate_X, train_plus_validate_y)
#     #print "		model_ix:{0}; model:{1}; score:{2}".format(model_ix, model, model.score(train_X, train_y))
#     if hasattr(model, "coef_"):
#         feat_importances = model.coef_[0]
#     elif hasattr(model, "feature_importances_"):
#         feat_importances = model.feature_importances_
#     elif isinstance(model, dummy.DummyClassifier):
#         feat_importances = [0 for i in range(len(features))]
#     else:
#         print "fatal error: feature importances for model:{0} unknown".format(model_ix)
#         exit(1)
#
#     models.ix[model_ix, 'feature_importances'] = str(sorted(zip(features, feat_importances),
#                                                             key=lambda
#                                                             p_val: p_val[1]
#                                                             #p_val: abs(p_val[1])	# abs for Linear SVC
#                                                             , reverse=True))
#     pred_test_y = model.predict(test_X)
#     pred_test_yM[model_ix] = pred_test_y
#     models.ix[model_ix, 'mislabels']				= (test_y != pred_test_y).sum()
#     models.ix[model_ix, 'f1_score']                 = metrics.f1_score(test_y, pred_test_y)
#     models.ix[model_ix, 'roc_auc_score']            = metrics.roc_auc_score(test_y, pred_test_y)
#     models.ix[model_ix, 'zero_one_loss']            = metrics.zero_one_loss(test_y, pred_test_y)
#     models.ix[model_ix, 'accuracy_score']           = metrics.accuracy_score(test_y, pred_test_y)
#     models.ix[model_ix, 'average_precision_score']  = metrics.average_precision_score(test_y, pred_test_y)
#     models.ix[model_ix, 'precision_score']  		= metrics.precision_score(test_y, pred_test_y)
#     models.ix[model_ix, 'recall_score']  			= metrics.recall_score(test_y, pred_test_y)
#     confusions = metrics.confusion_matrix(test_y, pred_test_y)
#     print "		model_ix:{0}; confusions:".format(model_ix)
#     pp.pprint(confusions)
#     print "		model_ix:{0}; classification:".format(model_ix)
#     print metrics.classification_report(test_y, pred_test_y)
#     #pdb.set_trace()
#
# pp.pprint(models)
# """
#
# ### 08.2	fit on stratified shuffled sample
# def myfit_stratified_samples(models, train_entity, validate_entity, train_plus_validate_entity, test_entity,
#                              features, pred_varname, mypred_varname, exp_prefix):
#
#     from sklearn import metrics
#     import pprint
#     pp = pprint.PrettyPrinter(indent=4)
#     import copy
#     import myplots
#     import matplotlib.pyplot as plt
#
#     models_cols_floats = 	['score'			# model
#                             ,'mislabels'		# key metric
#                                                 # metrics module
#                             ,'f1_score', 'precision_score', 'recall_score', 'roc_auc_score', 'zero_one_loss'
#                             ,'accuracy_score', 'average_precision_score',
#                             ]
#     for i in range(len(models_cols_floats)):
#         models[models_cols_floats[i]] = float(0.0)
#
#     models['feature_importances'] = ""
#
#     train_X = train_entity[features].values
#     train_y = train_entity[pred_varname].values
#     validate_X = validate_entity[features].values
#     validate_y = validate_entity[pred_varname].values
#     train_plus_validate_X = train_plus_validate_entity[features].values
#     train_plus_validate_y = train_plus_validate_entity[pred_varname].values
#     test_X = test_entity[features].values
#     test_y = test_entity[pred_varname].values
#     for model_ix, model_row in models.iterrows():
#         model = model_row['model']
#         model.fit(train_plus_validate_X, train_plus_validate_y)
#         models.ix[model_ix, 'score'] = model.score(train_plus_validate_X, train_plus_validate_y)
#         #print "		model_ix:{0}; model:{1}; score:{2}".format(model_ix, model, model.score(train_X, train_y))
#         if hasattr(model, "coef_"):
#             feat_importances = model.coef_[0]
#         elif hasattr(model, "feature_importances_"):
#             feat_importances = model.feature_importances_
#         else:
#             feat_importances = [0.0 for i in range(len(features))]
#
#         feat_importances_lst = sorted(zip(features, feat_importances),key=lambda p_val:
#                                             p_val[1]
#                                             #abs(p_val[1])	# abs for Linear SVC
#                                     , reverse=True)
#         models.ix[model_ix, 'feature_importances'] = str(["({0}, {1:0.4f})".format(tuple[0], tuple[1])
#                                                           for tuple in feat_importances_lst])
#
#     # add cross-validation score for entity / train_plus_validate_entity / train_entity ?
#         pred_test_y = model.predict(test_X)
#         test_entity[model_ix + mypred_varname] = pred_test_y
#         models.ix[model_ix, 'mislabels']				= (test_y != pred_test_y).sum()
#         models.ix[model_ix, 'f1_score']                 = metrics.f1_score(test_y, pred_test_y)
#         models.ix[model_ix, 'precision_score']  		= metrics.precision_score(test_y, pred_test_y)
#         models.ix[model_ix, 'recall_score']  			= metrics.recall_score(test_y, pred_test_y)
#         models.ix[model_ix, 'roc_auc_score']            = metrics.roc_auc_score(test_y, pred_test_y)
#         models.ix[model_ix, 'zero_one_loss']            = metrics.zero_one_loss(test_y, pred_test_y)
#         models.ix[model_ix, 'accuracy_score']           = metrics.accuracy_score(test_y, pred_test_y)
#         models.ix[model_ix, 'average_precision_score']  = metrics.average_precision_score(test_y, pred_test_y)
#         confusions = metrics.confusion_matrix(test_y, pred_test_y)
#         print "		model_ix:{0}; confusions:".format(model_ix)
#         pp.pprint(confusions)
#         print "		model_ix:{0}; classification:".format(model_ix)
#         print metrics.classification_report(test_y, pred_test_y)
#
#     #pp.pprint(models)
#     print "{0}".format(models.to_string(float_format=lambda x: '%0.4f' % x))
#
#     # separate plots for training & test errors ?
#     scatter_plot_columns = copy.copy(features)
#     #scatter_plot_columns.append(pred_varname)
#     scatter_plot_entity = test_entity[scatter_plot_columns]
#     for model_ix, model_row in models.iterrows():
#         model = model_row['model']
#         model_predict_y = model.predict(scatter_plot_entity[features].values)
#         myplots.myscatter_matrix(scatter_plot_entity,
#                                  pred_values=test_entity[pred_varname], mypred_values=model_predict_y,
#                                  alpha=0.2, diagonal='hist')
#         #plt.show()
#         exp_filename = exp_prefix + model_ix + "_test_scatter_entity" + '.png'
#         print "		exporting plot:{0} ...".format(exp_filename)
#         plt.savefig(exp_filename, dpi=200)
#
#     return models
#
# ### 08.3    fit on cross-validated samples
# def mybuild_models_df(models_df, k):
#
#     import copy
#
#     new_models_df = copy.copy(models_df)
#     models_cols_floats = 	['fit_score'		# model
#                             ,'fit_n', 'fit_mislabels'
#                             ,'predict_n'
#                             ,'f1_score', 'mislabels'
#                             ,'accuracy_score', 'average_precision_score'
#                             ,'precision_score', 'recall_score', 'roc_auc_score', 'zero_one_loss'
#                             ]
#     for i in range(len(models_cols_floats)):
#         new_models_df[models_cols_floats[i]] = float(0.0)
#
#     new_models_df['feature_importances'] = ""
#
#     #	ensure different dataframes for each k-item
#     new_models_df['fit_score'] = [(k * models_df.shape[0] + model_ix) for model_ix in range(models_df.shape[0])]
#     return new_models_df
#
# def myfit_cv_samples(n_folds, models_df, entity_df, cv_ix_varname
#                     ,features, pred_varname, mypred_varname, exp_prefix, tm_start
# 					,plt_scatter_matrix=True, plt_show=False):
#
#     from sklearn import metrics
#     import pprint
#     pp = pprint.PrettyPrinter(indent=4)
#     import copy
#     import myplots
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import math
#     import numpy as np
#     import datetime as tm
#
#     ###			create models Panel where:
#     ###				items 		= DataFrame of models for k-fold 	(axis = 0)
#     ###				major_axis	= model type		                (axis = 1)
#     ###				minor_axis 	= model & stats			            (axis = 2)
#
#     kf_items = ['cv_' + str(k) for k in range(1, n_folds+1)]
#     models_df_dict = dict(zip(kf_items,
#                           [mybuild_models_df(pd.DataFrame({'model': models_df['model'].values}
#                                                             , index=models_df.index)
#                                              ,k)
#                            for k in range(n_folds)]))
#     models_panel = pd.Panel(models_df_dict, items=kf_items)
#     if isinstance(entity_df, pd.SparseDataFrame):
#         entity_df = entity_df.to_dense()
#
#     for cv_fold, k_item in enumerate(kf_items):
#         print "\n[{0}]		cv_fold = {1}".format(str(tm.datetime.now() - tm_start), cv_fold)
#         validate_index = (cv_fold + 2) if (cv_fold + 2) <= n_folds else 1
#         #print "cv_fold = %d; validate_index = %d" % (cv_fold, validate_index)
#         train_plus_validate_entity_df =	entity_df[entity_df[cv_ix_varname] != cv_fold+1]
#         test_entity_df = 				entity_df[entity_df[cv_ix_varname] == cv_fold+1]
#         train_entity_df = 		train_plus_validate_entity_df[train_plus_validate_entity_df[cv_ix_varname]
#                                                                 != validate_index]
#         validate_entity_df = 	train_plus_validate_entity_df[train_plus_validate_entity_df[cv_ix_varname]
#                                                                 == validate_index]
#
#         train_X = train_entity_df[features].values
#         train_y = train_entity_df[pred_varname].values
#         validate_X = validate_entity_df[features].values
#         validate_y = validate_entity_df[pred_varname].values
#         train_plus_validate_X = train_plus_validate_entity_df[features].values
#         train_plus_validate_y = train_plus_validate_entity_df[pred_varname].values
#         test_X = test_entity_df[features].values
#         test_y = test_entity_df[pred_varname].values
#         models_df = models_panel[k_item]
#         for model_ix, model_row in models_df.iterrows():
#             model = model_row['model']
#             model.fit(train_plus_validate_X, train_plus_validate_y)
#             models_df.ix[model_ix, 'fit_score'] = model.score(train_plus_validate_X, train_plus_validate_y)
#             #print "		model_ix:{0}; model:{1}; score:{2}".format(model_ix, model, model.score(train_X, train_y))
#             models_df.ix[model_ix, 'fit_n'] = len(train_plus_validate_X)
#             models_df.ix[model_ix, 'fit_mislabels'] = (train_plus_validate_y !=
#                                                         model.predict(train_plus_validate_X)).sum()
#             if hasattr(model, "coef_"):
#                 feat_importances = model.coef_[0]
#             elif hasattr(model, "feature_importances_"):
#                 feat_importances = model.feature_importances_
#             else:
#                 feat_importances = [0.0 for i in range(len(features))]
#
#             feat_importances_lst = sorted(zip(features, feat_importances),key=lambda p_val:
#                                                 p_val[1]
#                                                 #abs(p_val[1])	# abs for Linear SVC
#                                         , reverse=True)
#             models_df.ix[model_ix, 'feature_importances'] = str(["({0}, {1:0.4f})".format(tuple[0], tuple[1])
#                                                               for tuple in feat_importances_lst])
#
#             # add cross-validation score for entity / train_plus_validate_entity_df / train_entity_df ?
#             pred_test_y = model.predict(test_X)
#             test_entity_df[model_ix + mypred_varname] = pred_test_y
#             models_df.ix[model_ix, 'predict_n']                 = len(test_y)
#             models_df.ix[model_ix, 'f1_score']                 	= metrics.f1_score(test_y, pred_test_y)
#             models_df.ix[model_ix, 'mislabels']					= (test_y != pred_test_y).sum()
#             models_df.ix[model_ix, 'accuracy_score']           	= metrics.accuracy_score(test_y, pred_test_y)
#             models_df.ix[model_ix, 'average_precision_score']	= metrics.average_precision_score(test_y, pred_test_y)
#             models_df.ix[model_ix, 'precision_score']  			= metrics.precision_score(test_y, pred_test_y)
#             models_df.ix[model_ix, 'recall_score']  			= metrics.recall_score(test_y, pred_test_y)
#             models_df.ix[model_ix, 'roc_auc_score']            	= metrics.roc_auc_score(test_y, pred_test_y)
#             models_df.ix[model_ix, 'zero_one_loss']            	= metrics.zero_one_loss(test_y, pred_test_y)
#             confusions = metrics.confusion_matrix(test_y, pred_test_y)
#             print "[{0}]			model_ix:{1}; confusions:".format(str(tm.datetime.now() - tm_start) + ";" + str(tm.datetime.now())
#                                                                         ,model_ix)
#             pp.pprint(confusions)
#             #print "		model_ix:{0}; classification:".format(model_ix)
#             #print metrics.classification_report(test_y, pred_test_y)
#             if (cv_fold == len(kf_items) - 1) and plt_scatter_matrix:
#                 myplots.myscatter_matrix(test_entity_df[features],
#                                         pred_values=test_y, mypred_values=pred_test_y, clf=model,
#                                         alpha=0.2, diagonal='density')
#                 if plt_show:
#                     plt.show()
#
#                 if exp_prefix is not None:
#                     exp_filename = exp_prefix + "cv%d_" % (cv_fold + 1) + model_ix + "_test_scatter_entity" + '.png'
#                     print "		exporting plot:{0} ...".format(exp_filename)
#                     plt.savefig(exp_filename, dpi=200)
#
#             if (cv_fold == len(kf_items) - 1):
#                 feat_importances_df = pd.DataFrame([tuple[1] for tuple in feat_importances_lst]
#                                                    , index=[tuple[0] for tuple in feat_importances_lst]
#                                                    , columns=['importance'])
#                 exp_filename = exp_prefix + "cv%d_" % (cv_fold + 1) + model_ix + "_feat_importances" + '.csv'
#                 print "		exporting data:{0} ...".format(exp_filename)
#                 feat_importances_df.to_csv(exp_filename, header=True, index_label='feature')
#                 test_entity_df['pred_' + glb_predict_varname] = pred_test_y
#                 if hasattr(model, "predict_proba"):
#                     test_entity_df['pred_proba_' + glb_predict_varname] = model.predict_proba(test_X)
#                 else:
#                     test_entity_df['pred_proba_' + glb_predict_varname] = -1.0
#
#                 srtd_test_entity_df = test_entity_df.sort([glb_predict_varname
#                                                         , 'pred_' + glb_predict_varname
#                                                         , 'pred_proba_' + glb_predict_varname])
#                 fltr_test_entity_df = srtd_test_entity_df[np.logical_and(
#                       srtd_test_entity_df[glb_predict_varname] == 0
#                     , srtd_test_entity_df['pred_' + glb_predict_varname] == 0)]
#                 ddive_test_entity_df = fltr_test_entity_df[:10]
#                 fltr_test_entity_df = srtd_test_entity_df[np.logical_and(
#                       srtd_test_entity_df[glb_predict_varname] == 0
#                     , srtd_test_entity_df['pred_' + glb_predict_varname] == 1)]
#                 ddive_test_entity_df = pd.concat(ddive_test_entity_df
#                                                  , fltr_test_entity_df[:-10])
#                 fltr_test_entity_df = srtd_test_entity_df[np.logical_and(
#                       srtd_test_entity_df[glb_predict_varname] == 1
#                     , srtd_test_entity_df['pred_' + glb_predict_varname] == 0)]
#                 ddive_test_entity_df = pd.concat(ddive_test_entity_df
#                                                  , fltr_test_entity_df[:10])
#                 fltr_test_entity_df = srtd_test_entity_df[np.logical_and(
#                       srtd_test_entity_df[glb_predict_varname] == 1
#                     , srtd_test_entity_df['pred_' + glb_predict_varname] == 1)]
#                 ddive_test_entity_df = pd.concat(ddive_test_entity_df
#                                                  , fltr_test_entity_df[:-10])
#                 exp_filename = exp_prefix + "cv%d_" % (cv_fold + 1) + model_ix + "_ddive" + '.csv'
#                 print "		exporting data:{0} ...".format(exp_filename)
#                 ddive_test_entity_df.to_csv(exp_filename, header=True, index_label='obs_id')
#
#
#         print "{0}".format(models_df.to_string(float_format=lambda x: '%0.4f' % x
#                                                 ,formatters={'fit_n': 			myint_with_commas
#                                                             ,'fit_mislabels': 	myint_with_commas
#                                                             ,'mislabels': 		myint_with_commas
#                                                             }))
#
#     ###			create cv Panel where:
#     ###				items 		= models	 			(axis = 0)
#     ###				major_axis	= cv_fold				(axis = 1)
#     ###				minor_axis 	= train_err, test_err	(axis = 2)
#
#     cv_df_dict = dict(zip(models_df.index,
#                           [pd.DataFrame({'train_n'	: models_panel.iloc[:,0].ix['fit_n']
#                                         ,'train_err': [0] * n_folds
#                                         ,'test_n' 	: models_panel.iloc[:,0].ix['predict_n']
#                                         ,'test_err' : [0] * n_folds
#                                         }
#                                         , index=kf_items)
#                            for model in range(len(models_df.index))]))
#     cv_panel = pd.Panel(cv_df_dict, items=models_df.index)
#
#     for model_ix, model in enumerate(models_df.index):
#         this_model_df = cv_panel[model]
#         cv_panel.ix[model,:,'train_err'] = models_panel.ix[:,model,'fit_mislabels']
#         cv_panel.ix[model,:,'test_err'] = models_panel.ix[:,model,'mislabels']
#         print "model: %s" % model
#         #pp.pprint(cv_panel[model])
#         print "{0}".format(cv_panel[model].to_string(float_format=lambda x: '%0.4f' % x
#                                                     ,formatters={'test_err'	: myint_with_commas
#                                                                 ,'test_n'	: myint_with_commas
#                                                                 ,'train_err': myint_with_commas
#                                                                 ,'train_n'	: myint_with_commas
#                                                                 }))
#
#     cv_error_df = pd.DataFrame({'cv_train_error'	: [0.0] * len(models_df.index)
#                                ,'mean_train_error'	: [0.0] * len(models_df.index)
#                                ,'var_train_error'	: [0.0] * len(models_df.index)
#                                ,'se_cv_train_error'	: [0.0] * len(models_df.index)
#                                #,'train_n'			: [0.0] * len(models_df.index)
#                                ,'cv_test_error'		: [0.0] * len(models_df.index)
#                                ,'mean_test_error'	: [0.0] * len(models_df.index)
#                                ,'var_test_error'	: [0.0] * len(models_df.index)
#                                ,'se_cv_test_error'	: [0.0] * len(models_df.index)
#                                #,'test_n'			: [0.0] * len(models_df.index)
#                                }, index=models_df.index)
#     for model_ix, model in enumerate(models_df.index):
#         cv_error_df.ix[model, 'cv_test_error'] 		= np.average(cv_panel.ix[model,:,'test_err']
#                                                             , weights=	[n_k * 1.0 /
#                                                                         sum(cv_panel.ix[model,:,'test_n'])
#                                                                         for n_k in cv_panel.ix[model,:,'test_n']])
#         cv_error_df.ix[model, 'mean_test_error'] 	= np.average(cv_panel.ix[model,:,'test_err'])
#         cv_error_df.ix[model, 'var_test_error'] 	= np.var(cv_panel.ix[model,:,'test_err'])
#         cv_error_df.ix[model, 'se_cv_test_error'] 	= math.sqrt(np.var(cv_panel.ix[model,:,'test_err']) * n_folds / (n_folds - 1))
#         #cv_error_df.ix[model, 'test_n']				= np.sum(cv_panel.ix[model,:,'test_n']) / n_folds
#         cv_error_df.ix[model, 'cv_train_error'] 	= np.average(cv_panel.ix[model,:,'train_err']
#                                                             , weights=	[n_k * 1.0 /
#                                                                         sum(cv_panel.ix[model,:,'train_n'])
#                                                                         for n_k in cv_panel.ix[model,:,'train_n']])
#         cv_error_df.ix[model, 'mean_train_error'] 	= np.average(cv_panel.ix[model,:,'train_err'])
#         cv_error_df.ix[model, 'var_train_error'] 	= np.var(cv_panel.ix[model,:,'train_err'])
#         cv_error_df.ix[model, 'se_cv_train_error'] 	= math.sqrt(np.var(cv_panel.ix[model,:,'train_err']) * n_folds / (n_folds - 1))
#         #cv_error_df.ix[model, 'train_n']			= np.sum(cv_panel.ix[model,:,'train_n']) / n_folds
#
#     print "cv_error_df:"
#     print "{0}".format(cv_error_df.to_string(float_format=lambda x: '%0.4f' % x
#                                                 ,formatters={
#                                                             }))
#
#     plt.figure()
#     #plot_cv_error_df.plot(kind='line', yerr=cv_error_df['se_cv_test_error'])
#     plt.errorbar(range(len(cv_error_df.index)), cv_error_df['cv_train_error'] / n_folds
#                 ,yerr=cv_error_df['se_cv_train_error']
#                 ,label='Train Error', marker='o')
#     plt.errorbar(range(len(cv_error_df.index)), cv_error_df['cv_test_error']
#                 ,yerr=cv_error_df['se_cv_test_error']
#                 ,label='Test Error', marker='o')
#     plt.xticks(range(len(cv_error_df.index)), list(cv_error_df.index), fontsize='x-small')
#     plt.xlim(-0.25, len(cv_error_df.index)-0.75)
#     """
#     plt.ylim(ymin=min(np.hstack((plot_cv_error_df['cv_train_error']
#                                 ,plot_cv_error_df['cv_test_error']))) -
#                   max(np.hstack((cv_error_df['se_cv_train_error']
#                                 ,cv_error_df['se_cv_test_error']))) -
#                   1
#             )
#     """
#     plt.legend(loc='best', shadow=True, fontsize='x-small')
#     plt.grid(b=True, axis='y', which='both', color='gray', linestyle='dashed')
#     #plt.set_axisbelow(True)
#     if plt_show:
#         plt.show()
#
#     if exp_prefix is not None:
#         exp_filename = exp_prefix + "cv_models" + '.png'
#         print "		exporting plot:{0} ...".format(exp_filename)
#         plt.savefig(exp_filename, dpi=200)
#
#
#     return cv_error_df
# """
#     # separate plots for training & test errors ?
#     scatter_plot_columns = copy.copy(features)
#     #scatter_plot_columns.append(pred_varname)
#     scatter_plot_entity = test_entity_df[scatter_plot_columns]
#     for model_ix, model_row in models_df.iterrows():
#         model = model_row['model']
#         model_predict_y = model.predict(scatter_plot_entity[features].values)
#         myplots.myscatter_matrix(scatter_plot_entity,
#                                  pred_values=test_entity_df[pred_varname], mypred_values=model_predict_y,
#                                  alpha=0.2, diagonal='hist')
#         #plt.show()
#         exp_filename = exp_prefix + model_ix + "_test_scatter_entity" + '.png'
#         print "		exporting plot:{0} ...".format(exp_filename)
#         plt.savefig(exp_filename, dpi=200)
#
#     return models_df
# """
#
# ###	09.	test model results  for k-fold0 test set
# ###	09.1	collect votes from each cross-validation for each model
# ### 09.2	collect votes from each model
# """ refactor as function
# del pred_test_yM[sel_models_index[0]]	# delete dummy model
#
# def mycollect_votes(pred_test_yM):
#     votes_pred_test_y = pred_test_yM.sum(axis=1).values
#     my_pred_test_y = copy.copy(votes_pred_test_y)
#     for i in range(len(votes_pred_test_y)):
#         if votes_pred_test_y[i] >= (pred_test_yM.shape[1] / 2):
#             my_pred_test_y[i] = 1
#         else:
#             my_pred_test_y[i] = 0
#
#     return (my_pred_test_y)
#
# my_pred_test_y = mycollect_votes(pred_test_yM)
# """
#
# ### 09.3 	export test data for inspection
# """ refactor as function
# exp_filename = exp_prefix + "tst.csv"
# entity_test[pred_varname] = test_y
# entity_test[pred_varname + '.predict'] = my_pred_test_y
# export_test = entity_test[entity_test[pred_varname] != entity_test[pred_varname + '.predict']]
# print "		exporting misclassifications in test data ({0} / {1} = {2}%) to file:{3} ...".format(
#                                                         export_test.shape[0],
#                                                         entity_test.shape[0],
#                                                         (float(export_test.shape[0]) / entity_test.shape[0]) * 100,
#                                                         exp_filename)
# export_test.to_csv(exp_filename)
# """
#
# ###	10.	predict results for new data
# ###	10.1	run models with data to predict
# """ refactor as function
# predict_X = predict[features].values
# predict_yM = pd.DataFrame(np.array([np.arange(predict_X.shape[0])] * len(sel_models_index)).T,
#                             #index=index,
#                             columns=sel_models_index)
# del predict_yM[sel_models_index[0]]	# delete dummy model
#
# for model_ix, model_row in models.iterrows():
#     model = model_row['model']
#     if isinstance(model, dummy.DummyClassifier):
#         continue
#
#     predict_yM[model_ix] = model.predict(predict_X)
# """
#
# ###	10.2	collect votes from each cross-validation for each model
# ### 10.3	collect votes from each model
# """ refactor as function
# my_predict_y = mycollect_votes(predict_yM)
# predict[pred_varname + '.predict'] = my_predict_y
# print "		prediction results:"
# mypred_var_count_print(predict, pred_varname + '.predict')
# """
#
# ###	11.	export results
# """ refactor as function
# submission = pd.DataFrame({ 'RefId' : predict.RefId, 'prediction' : my_predict_y })
# exp_filename = exp_prefix + "sub.csv"
# print "		exporting file:{0} ...".format(exp_filename)
# submission.to_csv(exp_filename)
# """
