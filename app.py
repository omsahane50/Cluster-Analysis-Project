import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


st.set_page_config(page_title="Global Development Indicators",  layout="wide")


with open("artifacts/pipeline.pkl", "rb") as f:

     pipeline = pickle.load(f)

country_mean = pd.read_csv("./artifacts/country_mean.csv")

cluster = pd.read_csv("./artifacts/cluster_formed_.csv")
cluster = cluster.rename(columns = {"lables_1":"Cluster label"})


cluster["Cluster label"] = np.where(cluster["Cluster label"] == 0, "Moderate Development Nation" , 
                                    np.where(cluster["Cluster label"] == 1, "High Development Nation" , 
                                             "Very High Development Nation"))


st.title("GLOBAL DEVELOPMENT INDICATORS(CLUSTER)")
st.subheader("Enter Values")

col1, col2, col3, col4 = st.columns([0.25,0.25,0.25,0.25])

birth_rate = col1.number_input("Birth Rate (0-1)",min_value=0, max_value=1 ) ### Birth Rate
business_taxrate = col2.number_input("Business Tax Rate") ### Business Tax Rate
co2_emmissions = col3.number_input("CO2 Emissions") ### CO2 Emissions
days_to_start_business = col4.number_input("Days To Start Business") ### Days to Start Business

energy_usage = col1.number_input("Energy Usage") ### Energy Usage
gdp = col2.number_input("GDP") ### GDP
health_per = col3.number_input("Health Exp % GDP") ### Health Exp % GDP
health_exp_capita = col4.number_input("Health Exp/Capita") ### Health Exp/Capita

hours_to_do_tax = col1.number_input("Hours to do Tax") ### Hours to do Tax
infant_mortality_rate = col2.number_input("Infant Mortality Rate") ### Infant Mortality Rate
internet_usage = col3.number_input("Internet Usage") ### Internet Usage
lending_intrest = col4.number_input("Lending Interest") ### Lending Interest

female = col1.number_input("Life Expectancy Female") ### Life Expectancy Female
male = col2.number_input("Life Expectancy Male") ### Life Expectancy Male
mobile_phone_usage = col3.number_input("Mobile Phone Usage") ### Mobile Phone Usage
population_0_14 = col4.number_input("Population 0-14") ### Population 0-14

population_15_64 = col1.number_input("Population 15-64") ### Population 15-64
population_65 = col2.number_input("Population 65+") ### Population 65+
population_total = col3.number_input("Population Total") ### Population Total
population_urban = col4.number_input("Population Urban") ### Population Urban

tourism_inbound = col1.number_input("Tourism Inbound") ### Tourism Inbound
tourism_outbound = col2.number_input("Tourism Outbound") ### Tourism Outbound
gdp_capita = col3.number_input("GDP/Capita") ### GDP/Capita
energy_capita = col4.number_input("Energy Usage/Capita") ### Energy Usage/Capita

_,col1, col2, col3,_ = st.columns([0.2,0.25,0.25,0.25,0.2])
co2_capita = col1.number_input("CO2 Emissions/Capita") ### CO2 Emissions/Capita
combined = col2.number_input("Life Expectancy Combined") ### Life Expectancy Combined
tourism_status = col3.selectbox("Tourism Status",np.unique(country_mean["Tourism Status"])) ### Tourism Status



prediction = ['Birth Rate', 'Business Tax Rate', 'CO2 Emissions',
       'Days to Start Business', 'Energy Usage', 'GDP', 'Health Exp % GDP',
       'Health Exp/Capita', 'Hours to do Tax', 'Infant Mortality Rate',
       'Internet Usage', 'Lending Interest', 'Life Expectancy Female',
       'Life Expectancy Male', 'Mobile Phone Usage', 'Population 0-14',
       'Population 15-64', 'Population 65+', 'Population Total',
       'Population Urban', 'Tourism Inbound', 'Tourism Outbound', 'GDP/Capita',
       'Energy Usage/Capita', 'CO2 Emissions/Capita',
       'Life Expectancy Combined', 'Tourism Status']


new_row = [birth_rate, business_taxrate, co2_emmissions, days_to_start_business, energy_usage, gdp, health_per,
           health_exp_capita, hours_to_do_tax, infant_mortality_rate, internet_usage,lending_intrest, female, male 
           ,mobile_phone_usage,population_0_14, population_15_64, population_65, population_total, population_urban,
           tourism_inbound, tourism_outbound, gdp_capita, energy_capita, co2_capita, combined, tourism_status]
values = pd.DataFrame([new_row], columns =prediction)

button = st.button("Predict Cluster")

if button:
    label = pipeline.predict(values)[0] 
      
    if label  == 0:
        label_name = "Moderate Development Nation"
        st.subheader(label_name)
        st.caption('GDP (Gross Domestic Product): Moderate GDP suggests a stable economic output.')
        st.caption('CO2 Emissions: Moderate CO2 emissions indicate a balanced environmental impact.')
        st.caption('Life Expectancy: Moderate life expectancy for both genders implies a satisfactory healthcare system.')
        st.caption('Internet Usage: Moderate internet usage indicates a reasonable level of technological connectivity.')
        st.caption('Inbound and Outbound Tourism: Moderate tourism suggests a stable level of international interaction.')
        st.caption('Countries Similar to the Input Country values are:')
        st.write(', '.join(cluster[cluster['Cluster label'] == label_name]["Country"].head()), ',etc')

        

    
    if label  == 1:
        label_name = "High  Development Nation"
        st.subheader(label_name)
        st.caption('GDP (Gross Domestic Product): Low GDP suggests an economy in the early stages of development.')
        st.caption('Business Tax Rate: High business tax rate indicates potential government revenue sources.')
        st.caption('Internet Usage: High internet usage suggests a focus on technological advancement.')
        st.caption('Urban Population: High urban population indicates significant urbanization.')
        st.caption('Inbound and Outbound Tourism: High tourism suggests growing international interactions.')
        st.caption('Countries to the Input Country values are:')
        st.write(', '.join(cluster[cluster['Cluster label'] == label_name]["Country"].head()), ',etc')

    
    if label  == 2:
        label_name = "Very High Development Nation"
        st.subheader(label_name)
        st.caption('GDP (Gross Domestic Product): Very high GDP indicates a robust and advanced economy.')
        st.caption('CO2 Emissions: Very high CO2 emissions may be a result of high industrialization.')
        st.caption('Life Expectancy: High life expectancy indicates a strong healthcare system.')
        st.caption('Population Distribution Across Age Groups: Very high population distribution across age groups suggests demographic stability.')
        st.caption('Urban Population: High urban population indicates significant urbanization and development.')
        st.caption('Countries to the Input Country values are:', )
        st.write(', '.join(cluster[cluster['Cluster label'] == label_name]["Country"].head()), ',etc')
        


    st.write("For Further Aalysis refer table Below:")
    st.write(cluster.drop(["Country", "Tourism Status"], axis= 1).groupby("Cluster label").mean())







