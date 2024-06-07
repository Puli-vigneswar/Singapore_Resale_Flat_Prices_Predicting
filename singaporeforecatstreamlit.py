#singapore FLAT price prediction 

import streamlit as st 
import pickle 
import pandas as pd 
# extracting data frame 
def getdf():
    df=pd.read_csv("D:/singapore_flat_sale_prediction/singaporeflatsdataframe.csv")
    return df

#deploying the saved label encoders
with open('D:/singapore_flat_sale_prediction/RANDOMFORESTmodel.pkl', 'rb') as file:
    rfr_model = pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/monthlabelencoder.pkl', 'rb') as file:
    monthle=pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/townlabelencoder.pkl', 'rb') as file:
    townle=pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/flattypelabelencoder.pkl', 'rb') as file:
    flatyple=pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/blocklabelencoder.pkl', 'rb') as file:
    blockle=pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/streetlabelencoder.pkl', 'rb') as file:
    streetle=pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/flatrangelabelencoder.pkl', 'rb') as file:
    rangele=pickle.load(file)
with open ('D:/singapore_flat_sale_prediction/flatmodellabelencoder.pkl', 'rb') as file:
    flatmodelle=pickle.load(file)

st.title(":violet[SINGAPORE FLAT PRICE PREDICTION]")
#initiating the tabs
tab1,tab2,tab3=st.tabs(["HOME","PREDICTION","SKILLS TAKEAWAY"])
with tab1:
    st.header(":rainbow[SINGAPORE FLAT PRICE PREDICTION ]")
    st.write("This is an user friendly application with this we can perform the forecasting of resale price of flats based on the historical data ")
    
    st.write(" which is helpful for forecasting the resale price of a flat in singapore  It aims to assist both potential buyers and sellers in estimating the resale value of a flat. ")
    
    st.write(" The application will benefit both potential buyers and sellers in the Singapore housing market. Buyers can use the application to estimate resale prices and make informed decisions, while sellers can get an idea of their flat's potential market value. Additionally, the project demonstrates the practical application of machine learning in real estate and web development.")
    
    st.subheader(":orange[Prediction model information]")
    st.write("The model used for the prediction is :green[RANDOM FOREST REGRESSION MODEL]")
    st.write("The performance of the model is accurate which is achieved by considering the historical data of singapore ")
    st.write("the accuracy of the random forest model is :red[98%]")
    
    url = "https://github.com/Puli-vigneswar/Singapore_Resale_Flat_Prices_Predicting"
    st.markdown(f"Check out this link of github {url}", unsafe_allow_html=True)
    
    
    lkin="https://www.linkedin.com/in/puli-vigneshwar-541575238/?trk=opento_sprofile_details"
    st.markdown(f"Check out personal linked profile {lkin}", unsafe_allow_html=True)
    
    
with tab2:
    
    col1, col2= st.columns(2)
    st.header("prediction of flat prices")
    df=getdf()
    with col1:
        # select month 
        selected_date = st.date_input("Select a year and month")
        yym=str(selected_date)
        month=yym[0:7] 
        #select town 
        town=st.selectbox("select town",df["town"].unique())
        
        # street name 
        streetdf=df[df["town"]==town]
        streets=st.selectbox("select street",streetdf["street_name"])
        
        blocdf=streetdf[streetdf["street_name"]==streets]
        # block 
        blocks=blocdf["block"].unique()
        selblock=st.selectbox("select block",blocks)
        
        # storey range 
        range=st.selectbox("select storey range",blocdf["storey_range"].unique())
    with col2:
        # fas input range 
        minfas=df["floor_area_sqm"].min()
        maxfas=df["floor_area_sqm"].max()
        
        fas=st.text_input(f"enter floor area sqm between :red[minimum value {minfas} and maximum value {maxfas}]",86)
        selfas=int(fas)
        #flat type
        typefl=df["flat_type"].unique()
        selftype=st.selectbox("select the flat type",typefl)
        #flat model 
        motypes=blocdf["flat_model"].unique()
        flatmodel=st.selectbox("select flat model type",motypes)
        
        #lease commence date 
        lcy = st.number_input("Enter a lease commence year", min_value=1950, max_value=1980)
    
    sampledf={"month":[month],
              "town":[town],
              "flat_type":[selftype],
              "block":[selblock],
              "street_name":[streets],
              "storey_range":[range],
              "floor_area_sqm":[selfas],
              "flat_model":[flatmodel],
              "lease_commence_date":[lcy]} 
    newdf1=pd.DataFrame(sampledf) 
    newdf1["month"]=monthle.transform(newdf1["month"]) 
    newdf1["town"]=townle.transform(newdf1["town"]) 
    newdf1["flat_type"]=flatyple.transform(newdf1["flat_type"]) 
    newdf1["block"]=blockle.transform(newdf1["block"]) 
    newdf1["street_name"]=streetle.transform(newdf1["street_name"]) 
    newdf1["storey_range"]=rangele.transform(newdf1["storey_range"]) 
    newdf1["flat_model"]=flatmodelle.transform(newdf1["flat_model"]) 

    pred=rfr_model.predict(newdf1)
    st.markdown(" ")
    try:
        if st.button("PREDICTED VALUE"):
            st.success(f'predicted resale price -- :green[{pred[0]}]')
    except:
        pass
with tab3:
    st.header(":rainbow[Skills takeaway]")
    st.write(" A LOT OF INSIGHTS i came to learn with the singapore  flat price Forecasting capstone ") 
    st.write("     >     Exploratory Data Analysis ")
    st.write("     >     Machine learning  ")
    st.write("     >     PYTHON scripting  ")
    st.write("     >     sklearn  ")
    st.write("     >     web deployment ")
    st.write(" Thanks for the support guvi team")
    
    # end of code
    