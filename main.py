import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV



class classification():

    def __init__(self,file_name_train,file_name_test,select='entropy',
                 nan_values=0,interpolate='linear',k_fold=5):
        """
        Initialize the classifiers
        :param 1) file_name: name of csv file
                2) nan_value: drop nan value or not 0 means drop, 1 means interpolate
                3) interpolate: types of interpolate : linear, zero, slinear,
                quadratic, cubic, barycentric,  polynomial, spline, piecewise_polynomial
                k_fold= number of K fold for cross_validation
        """
        self.file_name_train = file_name_train
        self.file_name_test = file_name_test
        self.nan_values = nan_values
        self.interpolate = interpolate
        self.k_fold=k_fold
        self.select=select
        """Some of these classifiers are in the scikit-learn examples and tutorials
        you may find them in https://scikit-learn.org"""
        self.models = {'Nearest Neighbors':KNeighborsClassifier(n_neighbors=5),
                     'Linear SVM':SVC(kernel="linear", C=0.5),
                     'RBF SVM':SVC(gamma=2, C=1),

                     'Decision Tree': DecisionTreeClassifier(max_depth=5),
                     'Random Forest':RandomForestClassifier(max_depth=5, n_estimators=10, max_features="auto"),
                     'MLP':MLPClassifier(alpha=1, max_iter=1000),
                     'AdaBoost':AdaBoostClassifier(),
                     'Naive Bayes' :GaussianNB(),
                     'QDA':QuadraticDiscriminantAnalysis(),
                     'SGD':SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
                     'NearestCenter': NearestCentroid(),
                     'Gradient Boosting':GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3,
                                                                    random_state=0, loss='exponential'),
                     'Logistic Regression': LogisticRegression(random_state=0, solver='lbfgs',penalty='l2')}

    #'Gaussian Process':GaussianProcessClassifier(1.0 * RBF(1.0)),

    def proccess_train_data(self):
        """Read the csv file and return the  clean and processed DataFrame
        output: clean data frame
        """
        if self.nan_values==0:
            # Remove All Nan values
            data=pd.read_csv(self.file_name_train+".csv").dropna(axis=0, how='any')
            # change the currency to value
            #see all object categorial data
            #data.select_dtypes(include=['object']).head(10)
            data["x41"]=data["x41"].replace('[\$,]', '', regex=True).astype(float)
            data["x45"]=data["x45"].replace('[,\%]', '', regex=True).astype(float)
            data["x34"]=data["x34"].astype('category').cat.codes
            data["x35"] = data["x35"].astype('category').cat.codes
            data["x68"]=data["x68"].astype('category').cat.codes
            data["x93"] = data["x93"].astype('category').cat.codes
        else:
            data = pd.read_csv(self.file_name_train + ".csv")
            data["x41"] = data["x41"].replace('[\$,]', '', regex=True).astype(float)
            data["x45"] = data["x45"].replace('[,\%]', '', regex=True).astype(float)
            data["x34"] = data["x34"].astype('category').cat.codes
            data["x35"] = data["x35"].astype('category').cat.codes
            data["x68"] = data["x68"].astype('category').cat.codes
            data["x93"] = data["x93"].astype('category').cat.codes
            data = data.interpolate(method =self.interpolate, limit_direction ='both')

        return data,data["y"].values

    def proccess_test_data(self):
        data = pd.read_csv(self.file_name_test + ".csv")
        data["x41"] = data["x41"].replace('[\$,]', '', regex=True).astype(float)
        data["x45"] = data["x45"].replace('[,\%]', '', regex=True).astype(float)
        data["x34"] = data["x34"].astype('category').cat.codes
        data["x35"] = data["x35"].astype('category').cat.codes
        data["x68"] = data["x68"].astype('category').cat.codes
        data["x93"] = data["x93"].astype('category').cat.codes
        data = data.interpolate(method=self.interpolate, limit_direction='both')
        return data

    def scoring(self,Y_label):
        score_dict={'accuracy':make_scorer(accuracy_score),
                    'precision':'precision_macro',
                    'f1 score': 'f1_macro',
                    'entropy' :make_scorer(log_loss),
                    'recall score ': make_scorer(recall_score),
                    'mean square': make_scorer(mean_squared_error)
                    }
        return score_dict

    def model_evaluation(self,train_data,scores):
        """Evaluate which model has the better results in terms of scoring model in scoring function"""
        X_train=train_data.iloc[0:100,0:train_data.shape[1]-1]
        Y_label=train_data.iloc[0:100,train_data.shape[1]-1]
        seed=10

        eval_results = pd.DataFrame(columns=scores.keys(),index=self.models.keys())
        for key,value in self.models.items():
            # kfold = model_selection.KFold(n_splits=self.k_fold, random_state=seed)
            result=cross_validate(value, X_train, Y_label,
                                    scoring=scores, cv = self.k_fold,
                                    return_train_score = True)
            for metric in scores.keys():
                tmp = 'test_' + metric
                eval_results.loc[key, metric] = result[tmp].mean()
                print('For "' + key + ' ' + metric + '" is {} '.format(result[tmp].mean()))

        eval_results.to_csv('evaluation_results.csv')
        return eval_results

    def visual_model_evaluation(self,dataframe):

        for metric in dataframe.columns:
            models=dataframe.loc[:,metric]
            plt.barh(np.arange(models.shape[0]), models.values, align='center', alpha=0.5)
            plt.yticks(np.arange(models.shape[0]), models.index)
            plt.xlabel(str(metric))
            plt.title(str(metric)+' for different classifiers')
            plt.savefig(str(metric)+'_test_set.png')

    def select_best_models(self,dataframe):
        """selection is based on class given input"""
        # for simplicity we already know that it is based on Entropy
        dataframe.sort_values(by=self.select,ascending=False)
        best_models = [dataframe[self.select].index[0], dataframe[self.select].index[1]]
        return best_models

    def optimizehyperparameters(self,bestmodels,train_data):
        """In this part we want to optimize the hyper parameters for models"""
        model_parameters={'Nearest Neighbors':[{'n_neighbors':np.arange(1,11,dtype=int)}],
                     'Linear SVM':[{'kernel':['linear'],'C':np.linspace(0.1,1,10)}],
                     'RBF SVM':[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}],

                     'Decision Tree': [{'max_depth':np.arange(1,11,dtype=int),'min_samples_split':np.linspace(2,8,4).astype(int)}],
                     'Random Forest':[{'max_depth':np.arange(1,11,dtype=int),'n_estimators':np.arange(10,101,dtype=int)}],
                     'MLP':[{'alpha':np.linspace(0.1,1,10),'max_iter':np.arange(500,1001,500,dtype=int)}],
                     'AdaBoost':[{'n_estimators':np.arange(50,101,25,dtype=int),'learning_rate':[1e-3,1e-2,.5,1]}],
                     'Naive Bayes' :[{'priors':[None]}],
                     'QDA':[{'reg_param':[0, 0.05,0.1,0.5,1]}],
                     'SGD':[{'loss':['hinge', 'log', 'squared_hinge', 'perceptron'],
                             'penalty':['l2','l1','elasticnet']}],
                     'NearestCenter': [{'metric':'euclidean'}] ,
                     'Gradient Boosting': [{'loss':['deviance', 'exponential'],'learning_rate':[0.01,0.05,0.1,0.5]}],
                     'Logistic Regression':[{'solver':['newton-cg','lbfgs','liblinear'],
                                             'penalty': ['l2', 'l1', 'elasticnet'],'C':[0.1,.5,1]}]}
        X_train = train_data.iloc[0:100, 0:train_data.shape[1] - 1]
        Y_label = train_data.iloc[0:100, train_data.shape[1]-1 ]

        trained_model={}
        test_data = self.proccess_test_data()
        for model in bestmodels:
            trained_model[model]=GridSearchCV(self.models[model], model_parameters[model], cv=self.k_fold,
                         scoring=make_scorer(log_loss)).fit(X_train,Y_label)

            test_data['Y_prediction']= trained_model[model].predict(self.proccess_test_data())
            test_data.to_csv(model+'_result1'+'.csv')
            print(trained_model[model].best_params_)

        return trained_model

    def run(self):

        evaluation=self.model_evaluation(self.proccess_train_data()[0],self.scoring(self.proccess_train_data()[1]))
        self.visual_model_evaluation(evaluation)
        bestmodels = self.select_best_models(evaluation)
        self.optimizehyperparameters( bestmodels, self.proccess_train_data()[0])









if __name__ == '__main__':
    classifiers = classification(file_name_train='train',
                                 file_name_test='test',nan_values=0)
    classifiers.run()




a=1

