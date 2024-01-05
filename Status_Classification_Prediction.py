import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel("Copper_Set.xlsx")
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

df['material_ref']=df['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)
df.drop(['id','material_ref'],axis=1,inplace=True)
df['quantity tons']=df['quantity tons'].apply(lambda x: np.nan if x <=0 else x)
df['selling_price']=df['selling_price'].apply(lambda x: np.nan if x <=0 else x)
#handling missing values using median and mode.
#object data types using mode.
df['item_date'].fillna(df['item_date'].mode().iloc[0],inplace=True)
df['status'].fillna(df['status'].mode().iloc[0],inplace=True)
df['delivery date'].fillna(df['delivery date'].mode().iloc[0],inplace=True)

#numerical data using median.
df['quantity tons'].fillna(df['quantity tons'].median(),inplace=True)
df['customer'].fillna(df['customer'].median(),inplace=True)
df['country'].fillna(df['country'].median(),inplace=True)
df['application'].fillna(df['application'].median(),inplace=True)
df['thickness'].fillna(df['thickness'].median(),inplace=True)
df['selling_price'].fillna(df['selling_price'].median(),inplace=True)

df['item type'] = OrdinalEncoder().fit_transform(df[['item type']])
df1 = df[df['status'].isin(['Won','Lost'])]
le = LabelEncoder()
df1.status = le.fit_transform(df1[['status']])

X=df1.drop(['item_date','delivery date','status'],axis=1)
y=df1['status']
rfc=RandomForestClassifier()
X_train,X_eval,y_train,y_eval=train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
rfc.fit(X_train,y_train)

pickle.dump(rfc, open('Status_Classification_Model.pkl','wb'))