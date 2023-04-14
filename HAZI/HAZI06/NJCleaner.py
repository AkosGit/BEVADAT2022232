import numpy as np
import pandas as pd

"""
1.  Értelmezd az adatokat!!!
    A feladat megoldásához használd a NJ transit + Amtrack csv-t a moodle-ból.
    A NJ-60k az a megoldott. Azt fogom használni a modellek teszteléséhez, illetve össze tudod hasonlítani az eredményedet.    

2. Írj egy osztályt a következő feladatokra:  
     2.1 Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     2.2 Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba 
     2.3 Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el  
     2.4 Dobjuk el a from és a to oszlopokat, illetve azokat a sorokat ahol van nan és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el  
     2.5 A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el   
     2.6 Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el  
         4:00-7:59 -- early_morning  
         8:00-11:59 -- morning  
         12:00-15:59 -- afternoon  
         16:00-19:59 -- evening  
         20:00-23:59 -- night  
         0:00-3:59 -- late_night  
    2.7 A késéseket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0min <= x < 5min   --> 0  
         5min <= x          --> 1  
    2.8 Dobd el a felesleges oszlopokat 'train_id' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    2.9 Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'. 
    2.10 Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'

3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re. 
    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik (0), ha 5p <= x --> késik (1).
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.

    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.

    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp 

4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el. 
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!

HAZI-
    HAZI06-
        -NJCleaner.py
        -HAZI06.py

##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################
"""
class NJCleaner:
    def __init__(self,path:str):
        self.data=pd.read_csv(path) 
    def order_by_scheduled_time(self):
        return self.data.copy().sort_values(['scheduled_time'], ascending=True)
    def drop_columns_and_nan(self):
        data2=self.data.copy()
        return data2.drop(["from","to"], axis=1).dropna()
    def convert_date_to_day(self):
        data2 = self.data.copy()
        data2['date'] = pd.to_datetime(data2['date'])
        data2['day'] = data2['date'].dt.day_name()
        return data2.drop(['date'], axis=1)
    def convert_scheduled_time_to_part_of_the_day(self):
        def part_of_the_day(time_str):
            time = pd.Timestamp(time_str).time()
            if time >= pd.Timestamp('04:00').time() and time <= pd.Timestamp('07:59').time():
                return 'early_morning'
            elif time >= pd.Timestamp('08:00').time() and time <= pd.Timestamp('11:59').time():
                return 'morning'
            elif time >= pd.Timestamp('12:00').time() and time <= pd.Timestamp('15:59').time():
                return 'afternoon'
            elif time >= pd.Timestamp('16:00').time() and time <= pd.Timestamp('19:59').time():
                return 'evening'
            elif time >= pd.Timestamp('20:00').time() and time <= pd.Timestamp('23:59').time():
                return 'night'
            else:
                return 'late_night'       
        data2 = self.data.copy()
        data2['part_of_the_day'] = data2['scheduled_time'].apply(part_of_the_day)
        data2.drop(["scheduled_time"],axis=1)
        return data2
    def convert_delay(self):
        def DelayConvert(x):
            if x < 5:
                return 0
            return 1
         #0min <= x < 5min   --> 0  
         #5min <= x          --> 1
        data2=self.data.copy()
        data2["delay"]=data2["delay_minutes"].apply(DelayConvert) 
        return data2
    def drop_unnecessary_columns(self):
        data2=self.data.copy()
        return data2.drop(["train_id","actual_time","delay_minutes"],axis=1)
    def save_first_60k(self,path):
        first60=self.data.copy().head(60000)
        first60.to_csv(path, index=False)
    def prep_df(self, path):
        self.data=self.drop_columns_and_nan()
        self.data=self.order_by_scheduled_time()
        self.data=self.convert_date_to_day()
        self.data=self.convert_scheduled_time_to_part_of_the_day()
        self.data=self.convert_delay()
        self.data=self.drop_unnecessary_columns()
        self.data=self.save_first_60k(path)

#cleaner=NJCleaner("C:\\Users\\Akos\\Documents\\BEVADAT2022232\\HAZI\\HAZI06\\2018_03.csv")
#cleaner.prep_df("C:\\Users\\Akos\\Documents\\BEVADAT2022232\\HAZI\\HAZI06\\NJ.csv")
"""
4. feladat: el vannak dobva?
"""


