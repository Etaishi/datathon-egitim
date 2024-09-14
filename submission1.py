import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing,model_selection,linear_model
veriler = pd.read_csv('train.csv',low_memory=False)
veri2023= pd.read_csv('test_x.csv',low_memory=False)
rowsayisi=len(veriler)
#HANGİ IMPUTE STRATEJİSİNİ BERLİRLEYECEZ ?
nan_count_per_column = veriler.isna().sum()

degerlendirme_puani = veriler[['Degerlendirme Puani']]
#AÇIK UÇLU SORULAR ÇIKARIYORUZ
tablo = veriler[['Basvuru Yili','Cinsiyet','Universite Turu'
                 ,'Universite Kacinci Sinif']]

tablo['Universite Turu'] = tablo['Universite Turu'].str.lower()


#LABELING
le =preprocessing.LabelEncoder()
tablo['Universite Turu'] = le.fit_transform(tablo['Universite Turu'])
tablo['Cinsiyet'] = le.fit_transform(tablo['Cinsiyet'])
tablo['Universite Kacinci Sinif'] = le.fit_transform(tablo['Universite Kacinci Sinif'])

#IMPUTING
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(tablo)
imputedveriler = imputer.transform(tablo)
imputer.fit(degerlendirme_puani)
degerlendirme_puani = imputer.transform(degerlendirme_puani)

#train ve test dğeişkenleri

x_train,x_test,y_train,y_test = model_selection.train_test_split(imputedveriler,degerlendirme_puani,test_size=0.33,random_state=0)
sc=preprocessing.StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

#SON TAHMİN
sontablo = veri2023[['Basvuru Yili','Cinsiyet','Universite Turu',
                 'Universite Kacinci Sinif']]
sontablo['Universite Turu'] = sontablo['Universite Turu'].str.lower()
#LABELING
le =preprocessing.LabelEncoder()
sontablo['Universite Turu'] = le.fit_transform(sontablo['Universite Turu'])
sontablo['Cinsiyet'] = le.fit_transform(sontablo['Cinsiyet'])
sontablo['Universite Kacinci Sinif'] = le.fit_transform(sontablo['Universite Kacinci Sinif'])

#IMPUTING
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(sontablo)
imputedverilerson = imputer.transform(sontablo)

sontablo=sc.fit_transform(sontablo)

tahmin2 =lr.predict(sontablo)


                      

                      

                      

                      