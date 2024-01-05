import numpy as np
import streamlit as st
import re
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler

st.set_page_config(page_title="Industrial copper Modeling ",layout="wide")
st.title(":pushpin: Industrial Copper modelling Prediction Application")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.subheader("By Tejesh V M:blue_heart:")

tab1,tab2=st.tabs(["PREDICT SELLING PRICE ","PREDICT STATUS"])
with tab1:
    status_options = ['Won', 'Lost']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product = ['611993', '611728', '628112', '628117', '628377', '640400', '640405', '640665','611993', '929423819',
               '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    with st.form('Copper_Modelling_Prediction-Selling_price'):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            status = st.selectbox('Status', status_options, key=1)
            item_type = st.selectbox('Item Type', item_type_options, key=2)
            country = st.selectbox('Country', sorted(country_options), key=3)
            application = st.selectbox('Application', sorted(application_options), key=4)
            product_ref = st.selectbox('Product Reference', product, key=5)
        with col3:
            quantity_tons = st.text_input('Enter Quantity Tons (Min:611728 & Max:1722207579)')
            thickness = st.text_input('Enter thickness (Min:0.18 & Max:400)')
            width = st.text_input('Enter width(Min:1, Max:2990)')
            customer = st.text_input('customer ID (Min:12458, Max:30408185)')
            submit_button = st.form_submit_button(label='Predict Selling Price')
            st.markdown('''
            ''', unsafe_allow_html=True)

        flag = 0
        pattern = '^(?:\d+|\d*\.\d+)$'
        for i in [quantity_tons, thickness, width, customer]:
            if re.match(pattern, i):
                pass
            else:
                flag = 1
                break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write('please enter a valid number space not allowed')
        else:
            st.write('you have entered an invalid value: ', i)

    if submit_button and flag == 0:
    
       
        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                country, float(customer), int(product_ref), item_type, status]])
    
        #Encoding the categorical data
        le = LabelEncoder()

        new_status = le.fit_transform(new_sample[:, [7]])
        new_itype = le.fit_transform(new_sample[:, [8]])
        
        
        new_sample = np.append(new_sample[:, [0, 1, 2, 3, 4, 5, 6, ]], new_status) 
        new_sample = np.append(new_sample, new_itype).reshape(1,-1) # Reshape to convert np array to features and samples array
        #new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, ]], new_status, new_itype), axis=1)

       
        #Scaling the data
        SS = StandardScaler()
        
        
        SS.fit_transform(new_sample)

        ## Load the model from the pickled file and predict the output from the input data
        with open(r'Selling_Price_Regressor_Model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        new_pred = loaded_model.predict(new_sample)[0]
        
        st.write('## :green[Predicted selling price: ', np.exp(new_pred))
with tab2:
    with st.form('Copper_Modelling_Prediction-Status'):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            cquantity_tons = st.text_input('Enter Quantity Tons (Min:611728 & Max:1722207579)')
            cthickness = st.text_input('Enter thickness (Min:0.18 & Max:400)')
            cwidth = st.text_input('Enter width (Min:1, Max:2990)')
            ccustomer = st.text_input('customer ID (Min:12458, Max:30408185)')
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

        cflag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
            if re.match(pattern, k):
                pass
            else:
                cflag = 1
                break

    if csubmit_button and cflag == 1:
        if len(k) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", k)

    if csubmit_button and cflag == 0:
   
        #Encoding the categorical data
        c_le = LabelEncoder()
  

        new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(product_ref),
                                item_type]])
            
        new_itype = c_le.fit_transform(new_sample[:, [8]])


        new_sample = np.append(new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7, ]], int(new_itype)).reshape(1,-1) 
        
        
        #Scaling the data
        SS = StandardScaler()
        
        new_sample = np.array(SS.fit_transform(new_sample), dtype=np.float64)
        
   
        with open(r"Status_Classification_Model.pkl", 'rb') as file:
            c_loaded_model = pickle.load(file)
        
        
        new_pred = c_loaded_model.predict(new_sample)

        if new_pred == 1:
            st.write('## :green[The Status is Won] ')
        else:
            st.write('## :red[The status is Lost] ')

