import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
pd.set_option("display.width",700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
#for categoric variables --> bar graph
x = np.array([1,4,8,12,15])
y = np.array([2,8,14,6,23])
df = sns.load_dataset("titanic")
plt.plot(x,y)
plt.plot(x,y,color="yellow")
plt.plot(x,y,marker="H")
plt.plot(x,y,linestyle="dotted")
plt.plot(x,y,marker="X",color="purple",linestyle="dashed")
print(plt.show())

df["sex"].value_counts().plot(kind="bar",color="green")
print(plt.show())

# For numeric variables --> histogram and boxplot
plt.hist(df["age"],color="red")
print(plt.show())
plt.boxplot(df["fare"])
print(plt.show())
plt.boxplot(x)
print(plt.show())

#multiple lines:
x = np.array([1,3,5,7,9,11,13,15])
y = np.array([2,4,6,8,10,12,14,16])
plt.plot(x,marker="x",color="red",linestyle="dashdot")
plt.plot(y,marker=".",color="black",linestyle="dashed")
plt.grid()
print(plt.show())

#subplots
x = np.random.randint(1,100,10)
y = np.random.randint(1,100,10)
z = np.random.randint(1,100,10)

plt.subplot(1,3,1)
plt.plot(x,y,color="green",linestyle="dotted",marker="h")
plt.title("1. Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()

plt.subplot(1,3,2)
plt.plot(x,z,color="blue",linestyle="dashed",marker="X")
plt.title("2. Graph")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid()

plt.subplot(1,3,3)
plt.plot(y,z,color="pink",linestyle="dashdot",marker="s")
plt.title("3. Graph")
plt.xlabel("Y")
plt.ylabel("Z")
plt.grid()
print(plt.show())

print(df.head())
#make a graph that shows mean age by gender using by matplotlib
data = df.groupby("sex").agg({"age":"mean"})
data.plot(kind="bar",color=["purple","blue"])
plt.title("mean age by gender")
plt.xlabel("gender")
plt.ylabel("age")
plt.grid()
plt.xticks(rotation=0)
print(plt.show())
#make a graph that shows mean age by gender using by seaborn library
sns.barplot(x="sex",y="age",data=df,estimator="mean",palette=["purple","blue"])
plt.title("Mean age by gender")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.grid()
plt.xticks(rotation=0)
print(plt.show())

#make a graph that shows mean survived by gender using by matplotlib
data = df.groupby("sex").agg({"survived":"mean"})
data.plot(kind="bar",color="green")
plt.title("Mean survived by gender")
plt.xlabel("Gender")
plt.ylabel("Survived")
plt.grid()
plt.xticks(rotation=0)
print(plt.show())

#make a graph that shows mean survived by gender using by seaborn library
sns.barplot(x="sex",y="survived",data=df,estimator="mean",palette=["yellow","black"])
plt.title("Mean survived by gender")
plt.xlabel("Gender")
plt.ylabel("Survived")
plt.grid()
plt.xticks(rotation=0)
print(plt.show())


#make a graph that shows mean survived by gender and class using by matplotlib
data = df.groupby(["sex","class"]).agg({"survived":"mean"})
data.plot(kind="bar",color ="green")
plt.title("Average Survived by gender and class")
plt.xlabel("gender and class")
plt.ylabel("average survived")
plt.grid()
plt.xticks(rotation = 45)
print(plt.show())

#make a graph that shows mean survived by gender and class using by seaborn
sns.barplot(x="sex",y="survived",hue="class",data=df,estimator="mean",palette="viridis")
plt.title("Average Survived by gender and class")
plt.xlabel("gender and class")
plt.ylabel("average survived")
print(plt.show())

#use catplot to distribution of age according to classes
sns.catplot(data=df,x="class",y="age")
print(plt.show())
sns.barplot(x="cut",y="price",hue="color",data=df) #more detailed
print(plt.show())


#Histogram ve Yoğunluk: sayısal değişkenlerin dağılımı için kullanılır:
df =sns.load_dataset("diamonds")
print(df.head())
#Distplot (Histogram):
sns.distplot(df["price"], kde= False, color="red") #kde yoğunluğu temsil etmektedir
print(plt.show())
sns.displot(x="price",data=df, kde=False, color="pink") #bu şekilde de distplot oluşturulabilir
print(plt.show())
sns.distplot(df["price"], kde=False, color="blue", bins= 1000) #ilkine göre daha detaylandı (1000 sütuna bölmüş gibi)
print(plt.show())
#Distplot (Hist and kde):
sns.distplot(df["price"], kde=True, color="grey") #there is both histogram and intensity curve
print(plt.show())
#Distplot (kde):
sns.distplot(df["price"], kde=True, hist=False, color="green") #there is only intensity, not a histogram
print(plt.show())
#kdeplot (Fill under the kde curve):
sns.kdeplot(df["price"], shade=True)
print(plt.show())
#Hisplot:
sns.histplot(x="price", data=df, kde=False, color="yellow")
print(plt.show())


#titanic veri setini tanımlayınız
df = sns.load_dataset("titanic")
print(df.head())
#görev 2: kadın ve erkek yolcuların sayısını bulunuz
print(df["sex"].value_counts())
#görev 3: Her bir sütuna (değişkene) ait unique değerlerin sayısını bulunuz
print(df.nunique())
#görev 4: pclass değişkeninin unqiue değerlerinin sayısını bulunuz:
print(df["pclass"].nunique()) #unique değerlerinin sayısı
print(df["pclass"].unique()) #unique değerler
#Görev 5: pclass ve parch değişkenlerinin unqiue değerlerinin sayısını bulunuz
print(df[["pclass","parch"]].nunique())
#görev 6: Embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
print(df["embarked"].dtypes)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtypes)
#görev 7: Embarked değeri C olanların tüm bilgilerini gösteriniz.
print(df[df["embarked"] == "C"])
#görev 8: Embarked değeri S olmayanların tüm bilgilerini gösteriniz
print(df[df["embarked"] != "S"])
#Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
print(df[(df["age"] < 30) & (df["sex"] == "female")])
#Görev 10: Fare'i 500'den büyük veya yaşı 70'den büyük yolcuların bilgilerini gösteriniz:
print(df[(df["fare"] > 500) | (df["age"] > 70)])
#Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
print(df.isnull().sum())
#Görev 12: who değişkenini dataframe'den çıkarınız
df.drop("who",axis=1,inplace=True)
print("who" in df)
#GÖrev 13: deck değişkenindeki boş değerleri deck değişkeninin en çok tekrar eden değeri (mode) ile doldurunuz
print(df["deck"].isnull().sum().any())
print(df["deck"].mode())
df["deck"].fillna(df["deck"].mode()[0],inplace=True)
print(df["deck"].isnull().sum().any())
#Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurunuz
print(df["age"].isnull().sum().any())
print(df["age"].median())
df["age"].fillna(df["age"].median(), inplace=True)
print(df["age"].isnull().sum().any())
#Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımından sum, count ve mean değerlerini bulunuzu
print(df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]})) #1.way
print(df.pivot_table("survived","pclass","sex",aggfunc=["sum","count","mean"]))
#Görev 16: 30 yaşın altında olanlar 1.30 a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın
#Yazdığınız fonksiyonu kullanarak titanik verisetinde age_flag adında birdeğişken oluşturunuz. (apply ve lambda yapılarınıkullanınız
df["age_flag"] = df["age"].apply(lambda x: 1.30 if x<30 else 0)
print(df.head())
#Görev 17: seaborn kütüphanesi içerisinden tips ver isetini tanımlayınız
df = sns.load_dataset("tips")
print(df.head())
print(df.shape)
print(df.info())
#Görev 18:Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulun.
print(df.groupby("time").agg({"total_bill":["sum", "min", "max", "mean"]})) #1. way
print(df.pivot_table("total_bill","time", aggfunc=["sum", "min", "max", "mean"])) #2. way
#Görev 19: günlere ve time göre total_bill değerlerinin toplamını, min,max ve ortalamasını bulunuz.
print(df.groupby(["day", "time"]).agg({"total_bill":["sum", "min", "max", "mean"]})) #1. way
print(df.pivot_table("total_bill", "day", "time" ,aggfunc=["sum", "min", "max", "mean"])) #2. way
#görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını min max ve meaninin bul
print(df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill":["sum", "min", "max", "mean"]})) #1. way
print(df[(df["time"] == "Lunch") & (df["sex"] == "Female")].pivot_table(["total_bill", "tip"],"day",aggfunc=["sum", "min", "max", "mean"])) #2. way
#Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
print(df[(df["size"] <3) & (df["total_bill"] > 10)].head())
print(df[(df["size"] <3) & (df["total_bill"] > 10)]["total_bill"].mean())
#Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
print(df.head().sort_values("total_bill_tip_sum", ascending=False))
#Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni birdataframe'e atayınız
df_new = df.sort_values("total_bill_tip_sum", ascending=False).head(30)
print(df_new)
print(df_new.shape)

#visualize the number of male and female persons:
#with matplotlib library
df["sex"].value_counts().plot(kind="bar",color="red")
plt.title("number of person by genders")
plt.xlabel("sex")
plt.ylabel("number of person")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

#with seaborn library
sns.countplot(x="sex", data=df)
plt.title("number of person by genders")
plt.xlabel("sex")
plt.ylabel("number of person")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

#visualize the average age of male and female persons:
#with matplotlib library:
data = df.groupby("sex").agg({"tip":"mean"})
data.plot(kind="bar", color="pink")
plt.title("average age by gender")
plt.xlabel("gender")
plt.ylabel("tip")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())
#with seaborn library:
sns.barplot(x="sex", y="tip", data=df, estimator="mean")
plt.grid()
plt.title("average age by gender")
plt.xticks(rotation=45)
print(plt.show())

#visualize the average tip according to sex and time
#with matplotlib library:
data = df.groupby(["sex","time"]).agg({"tip":"mean"})
data.plot(kind="bar",color="purple")
plt.title("average tip by gender and time")
plt.xlabel("gender and time")
plt.ylabel("tip")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())
#with seaborn library:
sns.barplot(x="sex", y="tip", hue="time", data=df, estimator="mean", palette="viridis")
plt.grid()
plt.title("average tip by gender and time")
plt.xticks(rotation=45)
print(plt.show())

