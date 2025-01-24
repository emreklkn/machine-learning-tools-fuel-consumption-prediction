import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator , TransformerMixin ,RegressorMixin ,clone
# koddaki bazı bölümleri anlamayanlar için notlar bırakıyorum 
#coef bağımsız değişken
# bias :erçek hedef değişken ile tahmin edilen değerler arasındaki sistematik hata


#clone :Bir scikit-learn tahminleyicisinden (modelden) yeni bir kopya oluşturur, ancak bu kopya eğitilmemiştir.
# Baseestimator : Model parametrelerini kolayca ayarlamak veya almak için kullanılır.
#TransformerMixin : ham veriyi dönüştürerek analiz edilebilir hale getirir (örneğin, ölçekleme, normalizasyon). fit_transform metodunu sağlar
#RegressorMixin :Regresyon modelleri için score metodunu sağlar. Bu metot, modelin performansını ölçmek için R^2 skorunu döndürür.
# XGBoost
import xgboost as xgb

# warning kapatma
import warnings
warnings.filterwarnings('ignore')

# data yükleme
column_name = ["MPG", "Cylinders", "Displacement","Horsepower","Weight","Acceleration","Model Year", "Origin"]
data = pd.read_csv("auto-mpg.data", names = column_name, na_values = "?", comment = "\t",sep = " ", skipinitialspace = True)

data = data.rename(columns = {"MPG":"target"})# mil başına yakıt tüketimini target yaptık

print(data.head())
print("Data shape: ",data.shape)

data.info()

describe = data.describe()# data özelliğini verir mean std vs ...
# burada bakıldığında mean ve %50 aynı değil bu da skewness olduğunu doğrular


# %% eksik değer 
print(data.isna().sum())# Eksik değerlerin hangi sütunlarda ve ne kadar olduğunu görmek için kullanılır.

data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())#Eksik değerleri doldurmak, modelin bu sütunu kullanabilmesi için gereklidir. Ortalama ile doldurmak, yaygın bir stratejidir.
#horsepower = beygirgücü
print(data.isna().sum())#yeniden Eksik değerlerin hangi sütunlarda ve ne kadar olduğunu görmek için kullanılır.

sns.distplot(data.Horsepower)#Verinin dağılımını görselleştirmek, verinin normal dağılıma yakın olup olmadığını veya başka bir desen içerip içermediğini anlamak için kullanılır.

# sağa doğru pozitif skewness var 

# %% keşifsel veri analizi - eda
"""korelasyon matrixi çizdiriyoruz"""
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

# anlamları 0.75 den büyük olanları filtreliyoruz ve onları çıkarıyoruz
threshold = 0.75
filtre = np.abs(corr_matrix["target"])>threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()


"""
ilişkilere bakma
"""

sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()

"""
cylinders ve origin kategorik olabilir (feature engineering)
"""

plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

# outlier(aykırı değer) bakmak için box plota bakıcaz


for c in data.columns:# ıqr dan tespit ediyoruz hızlanma ve beygir gücünde aykırı değer var 
    plt.figure()
    sns.boxplot(x = c, data = data, orient = "v")


"""
outlier: horsepower ve acceleration da aykırı değerler var bunları çıkarmamız gerekiyor
"""







#%%aykırı değer çıkarımı
"""SIRADA AYKIRI DEĞER ÇIKARMA İŞLEMİ YAPACAĞIZ"""
"""q1 ve q3 tespiti şöyle yapılıyor describe da olan %75 = q3 oluyor , %25 ise q1 oluyor bunları 
describe da index olarak alıyoruz , yani tabloda ki 0. index count a karşılık geliyor 4. index %25 e 6.index ise %75 e denk geliyor"""
thr = 2 # 2 den fazla olanları tespit edicez , deneyreke değitştirebiliriz
horsepower_desc = describe["Horsepower"]# horsepower beygir gücü yani onda var aykırı değer
q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp
filter_hp_bottom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top

data = data[filter_hp]

acceleration_desc = describe["Acceleration"]# hız da var bide aykırı değer bunları ıq
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc # q3 - q1
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top= data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc] # remove Horsepower outliers




#%% feature engineering = özellik çıkarımı 
# skewness = data eğriliği yani sağa sola doğru yatıksa pozitif veya negatif skewness vardır buda oralarda aykırı değer olabilceği anlamına gelir
# bunları normal gaus dağılımına çevirmemiz gerekiyor 

# target dependent variable
sns.distplot(data.target, fit = norm)

(mu, sigma) = norm.fit(data["target"])
print("mu/(mean): {}, sigma(standart sapma) = {}".format(mu, sigma))


#şimdi dağılımın ne kadar gaus yani normal dağılım olup olmadığına bakmak için
#qqplot var ordan bakabiliyoruz
#
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()# detaylı baktığımızda datamız kırmızı çizgi üzerinde değil


#skewness lik yanı çarpıklığını azaltmak için log transforma sokacağız

data["target"] = np.log1p(data["target"]) 

plt.figure()
sns.distplot(data.target, fit = norm)

(mu, sigma) = norm.fit(data["target"])
print("mu: {}, sigma = {}".format(mu, sigma))# değerler değişmti

##qqplota bakalım tekrardan 
stats.probplot(data["target"], plot = plt)
plt.show()# detaylı baktığımızda datamız kırmızı çizgi üzerinde değil

"""düzelme olmuş"""

#bağımsız değişkenlerin "hız"gibi skewness değerlerine bakıcağız çarpıklık yani 

skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = ["skewed"])
#dropna dememiz lazım null değer olabilir

"""skewnessları düzeltmek için bir yöntem var ileriki çalışmalar için not bırakıyorum :Box Cox Transformation"""


#%% one hot encoding -- mesala silindirlerde 1, 2 ,3 diyeceğimize 0 0 1 , 100 , 010 şekline çeviriyoruz 
# yapma sebebimiz ise verilerde hata değeri mesala 3 silindirde 3 çıkıcak ama biz bunu one hot encoding yaptığımızda 1 çıkıcak

#72 ve 73 satırdaki kodları çalıştırdığımızda dikkat edersek cylinders ve origin kategorik olabilir bu yüzden kategorik verilerde one hot encoding yapacağız
data["Cylinders"] = data["Cylinders"].astype(str)  
data["Origin"] = data["Origin"].astype(str) 

data = pd.get_dummies(data)







#%% train test split 
# Split
x = data.drop(["target"], axis = 1)
y = data.target

test_size = 0.9 # verinin yüzde 10 ile eğitim yapılıcak yüzde 90 ile test edilicek normalde yüzde 80 train yüzde 20 test olmalı genelede
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42)


# Standardization
scaler = RobustScaler()  # RobustScaler #StandardScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)# fit dememizin sebebi zaten yukarıda ayarlanmıştı


# %% Regresyon Modelleri
# regresyon hatayı minimize edetmeyi amaçlar
#linear regresyon
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef: ",lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Linear Regression MSE: ",mse)

#0.02 loss değeri geldi

# Ridge Regresyon (L2)
# l2 = (LESSsquareror + Lambda*slope^2) = bunu minimaze etmeyi amaçlar
#bu overfittingi engeller
# ridge gereksiz bağımsız değişkenlere 0.0001 gibi çok düşük değer verir
# biraz bias var : erçek hedef değişken ile tahmin edilen değerler arasındaki sistematik hata
ridge = Ridge(random_state = 42, max_iter = 10000)
alphas = np.logspace(-4,-0.5,30)# en iyi alpha değerini arayacak -4 den -0.5 e kadar 30 adet oluşturucak

tuned_parameters = [{'alpha':alphas}] #parametre return ediyoruz
n_folds = 5
 #grid search parametre uzayı tarayarak en iyi parametreleri bulur , modelin performansını optimize etmeyi amaçlar
clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ",clf.best_estimator_.coef_)#en iyi parametrelere bakıyoruz
ridge = clf.best_estimator_
print("Ridge Best Estimator: ", ridge)# 

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Ridge MSE: ",mse)
print("-----------------------------------------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")


# Lasso Regression (L1)
# l1 = (LESSsquareror + Lambda*|slope|) = bunu minimaze etmeyi amaçlar
# ridgen farkı slope karesi değilde mutlak değer olarak minimize eder
# farkları# lasso gereksiz bağımsız değişkenlere 0 verir feature selection yapılabilir bunla
# biraz bias var : erçek hedef değişken ile tahmin edilen değerler arasındaki sistematik hata
# overfittingi engeller
lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)# en iyi alpha değerini arayacak -4 den -0.5 e kadar 30 adet oluşturucak

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Lasso Coef: ",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ",lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Lasso MSE: ",mse)
print("---------------------------------------------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")


# ElasticNet
# elastiknet hem ridge hemde lasso yu kullanıyor 
# elastiknet = (LESSsquareror + Lambda*|slope|) veya (LESSsquareror + Lambda*slope^2) ikisinide kullanarak hatayı minimize etmeyi amaçlıyor

parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, Y_train)


print("ElasticNet Coef: ",clf.best_estimator_.coef_)# bağımsız değişkenlerin en iyi parametreleri
print("ElasticNet Best Estimator: ",clf.best_estimator_)


y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("ElasticNet MSE: ",mse)


"""
StandardScaler
    Linear Regression MSE:  0.020632204780133015
    Ridge MSE:  0.019725338010801216
    Lasso MSE:  0.017521594770822522
    ElasticNet MSE:  0.01749609249317252
RobustScaler:
    Linear Regression MSE:  0.020984711065869643
    Ridge MSE:  0.018839299330570554
    Lasso MSE:  0.016597127172690837
    ElasticNet MSE:  0.017234676963922273  
"""
# %% XGBoost regresyon
#gradyan artırma (gradient boosting) algoritmasını hızlı, verimli ve optimize bir şekilde uygulayan popüler bir makine öğrenimi kütüphanesidir.
# Karar ağaçları (decision trees) üzerine kurulu bir modeldir ve sınıflandırma, regresyon ve sıralama problemlerinde sıklıkla kullanılır.




parametersGrid = {'nthread':[4], #hyperthread kullanıldığında xgboost yavaşlayabilir
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = 5, verbose=True)

clf.fit(X_train, Y_train)
model_xgb = clf.best_estimator_

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("XGBRegressor MSE: ",mse)

# %% Averaging Models : modellerin ortalamasını anlıyor
#belirli tahminleyiciler var ve bunların ortalamasını alıp test sonucu elde eder

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    # verilere uyacak şekilde orijinal modellerin kopyalarını tanımlarız
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Kopyalanmış temel modelleri eğitme
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Şimdi kopyalanmış modeller için tahminler yapıyoruz ve ortalamalarını alıyoruz
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)  


averaged_models = AveragingModels(models = (model_xgb, lasso))# yukarıdaki sınıfı çağırıp xgboost ve lassoyu kullanıyoruz
averaged_models.fit(X_train, Y_train)# oluşan yeni modeli öğrenmeye koyuyoruz

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Averaged Models MSE: ",mse)

"""
StandardScaler:
    Linear Regression MSE:  0.020632204780133015
    Ridge MSE:  0.019725338010801216
    Lasso MSE:  0.017521594770822522
    ElasticNet MSE:  0.01749609249317252
    XGBRegressor MSE: 0.017167257713690008
    Averaged Models MSE: 0.016034769734972223
RobustScaler:
    Linear Regression MSE:  0.020984711065869643
    Ridge MSE:  0.018839299330570554
    Lasso MSE:  0.016597127172690837
    ElasticNet MSE:  0.017234676963922273
    XGBRegressor MSE: 0.01753270469361755
    Averaged Models MSE: 0.0156928574668921
"""



