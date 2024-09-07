import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing,model_selection

#EKSİK VERİLERİ DOLDURMA 
veriler = pd.read_csv('veriler.csv')
eksik_veriler =pd.read_csv('eksikveriler.csv')
print(eksik_veriler)
print('-----------------')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#object instance
yas=eksik_veriler.iloc[:,1:4].values # iloc un virgülden önceki kısmı row, sonraki kısmı columnu ifade eder
imputer.fit(yas)#impute stratejisine göre missing value içeren columnlarda impute etmek üzere hesaplama yapıyor
yas=imputer.transform(yas)#yukarıda yapılan hesaplamayı missing valueların yerine koyuyor
#fit_transform ile yukarıdaki iki satır aynı anda işlenebilir.
print(yas)
#----------------------------------------------------------------------------------------------------
#DATA LABELING
#burada kategorik verileri sayısal veriye çeviriyoruz
ulke=veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)
ohe = preprocessing.OneHotEncoder()#onehot vektör, bir adet 1 ve diğer değerlerin 0 olduğu vektöre denir.
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
#----------------------------------------------------------------------------------------------------
#veri setlerini rastgele bölmek için önce dataframeleri oluşturuyoruz
sonuc = pd.DataFrame(data=ulke, index = range(22),columns = ['fr','tr','us'])
sonuc2 = pd.DataFrame(data=ulke, index = range(22), columns=['boy','kilo','yas'])
cinsiyet=veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
s=pd.concat([sonuc,sonuc2],axis=1)

#şimdi s dataframeini veri setlerine böleceğiz
x_train,x_test,y_train,y_test = model_selection.train_test_split(s,sonuc3,test_size=0.33,random_state=0)
#----------------------------------------------------------------------------------------------------

#featureları birbiriyle karşılaştırabilmek ölçeklendiriyoruz
sc = preprocessing.StandardScaler()
x_train =sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)







