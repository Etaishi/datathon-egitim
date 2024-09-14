import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn import preprocessing,model_selection,linear_model
veriler = pd.read_csv('train.csv',low_memory=False)
veri2023= pd.read_csv('test_x.csv',low_memory=False)
#HANGİ IMPUTE STRATEJİSİNİ BERLİRLEYECEZ ?

degerlendirme_puani = veriler[['Degerlendirme Puani']]
#AÇIK UÇLU SORULAR ÇIKARIYORUZ
tablo = veriler[['Basvuru Yili','Cinsiyet','Dogum Tarihi','Dogum Yeri','Universite Turu',
                 'Ikametgah Sehri','Universite Adi','Burs Aliyor mu?','Bölüm',
                 'Universite Kacinci Sinif','Universite Not Ortalamasi','Universite Kacinci Sinif',
                 'Universite Not Ortalamasi','Lise Sehir','Lise Turu','Lise Mezuniyet Notu',
                 'Baska Bir Kurumdan Burs Aliyor mu?','Baska Kurumdan Aldigi Burs Miktari',
                 'Anne Egitim Durumu','Anne Calisma Durumu','Anne Sektor','Baba Egitim Durumu',
                 'Baba Calisma Durumu','Baba Sektor','Kardes Sayisi','Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
                 'Profesyonel Bir Spor Daliyla Mesgul musunuz?','Ingilizce Biliyor musunuz?','Ingilizce Seviyeniz?',
                 'Aktif olarak bir STK üyesi misiniz?','Girisimcilikle Ilgili Deneyiminiz Var Mi?']]
#Doğum tarihi preprocess---------------------------------------------------------------------------------------------
def parse_date(date_str):
    try:
        if isinstance(date_str, str):
            parsed_date = parse(date_str, fuzzy=True)
            if parsed_date.year < 1950 or parsed_date.year > datetime.now().year:
                return None  # Set invalid dates to None (NaN)
            return parsed_date
        return None  # If date_str is not a string, return None
    except (ValueError, OverflowError):
        return None  # Set invalid dates to None (NaN)
# Parse birth dates
tablo['Dogum Tarihi'] = tablo['Dogum Tarihi'].apply(parse_date)
tablo['Basvuru Yili'] = pd.to_datetime(tablo['Basvuru Yili'],format='%Y')

def calculate_age(birth_date, apply_date):
    if birth_date is not None:
        return apply_date.year - birth_date.year
    return np.nan


tablo['Dogum Tarihi'] = tablo.apply(lambda row: calculate_age(row['Dogum Tarihi'], row['Basvuru Yili']), axis=1)
tablo['Dogum Tarihi'] = tablo['Dogum Tarihi'].where(tablo['Dogum Tarihi'] >= 18, np.nan)
mean_age = tablo['Dogum Tarihi'].mean()
mean_age = np.floor(mean_age)
tablo['Dogum Tarihi'].fillna(mean_age, inplace=True)

#-----------------------------------------------------------------------
#Doğum yeri ve ikametgah preprocess-----------------------------------------
sehirler = [
    "adana", "adiyaman", "afyonkarahisar", "agri", "aksaray", "amasyi", "ankara", "antalya",
    "ardahan", "artvin", "aydin", "balikesir", "bartin", "batman", "bayburt", "bilecik", "bingol",
    "bitlis", "bolu", "burdur", "bursa", "canakkale", "cankiri", "corum", "denizli", "diyarbakir",
    "duzce", "edirne", "elazig", "erzincan", "erzurum", "eskisehir", "gaziantep", "giresun", "gumushane",
    "hakkari", "hatay", "igdir", "isparta", "istanbul", "izmir", "kahramanmaras", "karabuk", "karaman",
    "kars", "kastamonu", "kayseri", "kilis", "kirikkale", "kirklareli", "kirsehir", "kocaeli", "konya",
    "kutahya", "malatya", "manisa", "mardin", "mersin", "mugla", "mus", "nevsehir", "nigde", "ordu",
    "osmaniye", "rize", "sakariya", "samsun", "siirt", "sinop", "sivas", "sanliurfa", "sirnak", "tekirdag",
    "tokat", "trabzon", "tunceli", "usak", "van", "yalova", "yozgat", "zonguldak"
]


# Function to replace Turkish characters with English equivalents
def replace_turkish_chars(text):
    if not isinstance(text, str):
        return None
    replacements = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
    }
    for turkish_char, english_char in replacements.items():
        text = text.replace(turkish_char, english_char)
    return text

# Function to extract the primary city name based on known cities
def extract_primary_city(text, sehirler):
    if not isinstance(text, str):
        return None
    else:
        text = replace_turkish_chars(text).lower()
        # Handle different delimiters
        if '/' in text:
            parts = text.split('/')
        elif '-' in text:
            parts = text.split('-')
        elif ',' in text:
            parts = text.split(',')
        else:
            parts = [text]
        parts = [part.strip() for part in parts]
        # Compare each part to the known cities list
        for part in parts:
            if part in sehirler:
                return part
        return None
tablo['Dogum Yeri'] = tablo['Dogum Yeri'].apply(lambda sehir: extract_primary_city(sehir, sehirler))
tablo['Ikametgah Sehri'] = tablo['Ikametgah Sehri'].apply(lambda sehir: extract_primary_city(sehir, sehirler))
#------------------------------------------------------------------------------------------------------------------------------------
#Universite türü preprocess
tablo['Universite Turu'] = tablo['Universite Turu'].astype(str).str.lower()
#----------------------------------------------------------------------------
#Cinsiyet preprocess---------------------------------------------------------
tablo['Cinsiyet']=tablo['Cinsiyet'].str.lower()
tablo['Cinsiyet'] = tablo['Cinsiyet'].replace('belirtmek istemiyorum', np.nan)
#-----------------------------------------------------------------------------
#Burs alıyor mu preprocess---------------------------------------------------------
tablo['Burs Aliyor mu?']=tablo['Burs Aliyor mu?'].str.lower()
#---------------------------------------------------------------------------------
#İngilizce biliyor musunuz preprocess---------------------------------------------------------
tablo['Ingilizce Biliyor musunuz?']=tablo['Ingilizce Biliyor musunuz?'].str.lower()
#---------------------------------------------------------------------------------

'''
#LABELING
le =preprocessing.LabelEncoder()
tablo['Cinsiyet'] = le.fit_transform(tablo['Cinsiyet'])
tablo['Universite Turu'] = le.fit_transform(tablo['Universite Turu'])
tablo['Universite Kacinci Sinif'] = le.fit_transform(tablo['Universite Kacinci Sinif'])


#train ve test dğeişkenleri

x_train,x_test,y_train,y_test = model_selection.train_test_split(tablo,degerlendirme_puani,test_size=0.33,random_state=0)
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
'''

                      

                      

                      

                      
