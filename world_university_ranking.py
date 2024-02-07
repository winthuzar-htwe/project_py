from cProfile import label
from curses import use_default_colors
from multiprocessing import Value
from xml.etree.ElementPath import get_parent_map
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pycountry
import altair as alt

st.set_page_config(
    page_title="World University Ranking",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

df = pd.read_csv('../Week08/World University Ranking.csv', index_col=0)
st.title("World University Ranking")
st.write(df)

with st.sidebar:
    st.title(' World University Ranking')
    
    year_list = list(df.Year.unique())[::-1]
    
    selected_year = st.selectbox('Select Year', year_list)
    df_selected_year = df[df.Year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="Student Population", ascending=False)

    country_list = list(df.Country.unique())
    selected_country_name = st.selectbox('Select Country Name', country_list)

uni_count = df.iloc[:12430,:].groupby(['Country', 'Country Code'], as_index = False)['University Name'].count()
fig = px.choropleth(uni_count, locations = 'Country Code', color = 'University Name', hover_name = 'Country')
st.markdown('#### Total Numbers of University by Country')
st.plotly_chart(fig, use_container_width=True)

count_10 = df.sort_values(['Rank'],ascending=False).groupby(['Country','Country Code','University Name']).head(10)
fig = px.bar(count_10, x = 'Country', y = 'Rank', color='Country Code')
st.markdown('#### University Ranking by Country')
st.plotly_chart(fig, use_container_width=True)

df['Students to Staff Ratio'] = df['Students to Staff Ratio'].astype(float)
pf = df.groupby(['Year','University Name'],as_index = False)['Students to Staff Ratio'].sum()
fig = px.bar(pf, x = 'Year', y='Students to Staff Ratio', color= 'Year')
st.markdown('#### Students to Staff Ratio by Year')
st.plotly_chart(fig, use_container_width=True)

top10 = df.loc[:10,['Teaching','Research Environment','Industry Impact','International Outlook']]
fig = plt.figure(figsize = (15,6))
sns.lineplot(data = top10)
st.markdown('#### University Research for Students')
st.pyplot(fig, use_container_width=True)

st.write(df.describe())

pf = df.drop(columns={'University Name','Country','International Students','Female to Male Ratio','Country Code'})
corr = pf.corr()
fig = plt.figure(figsize = (12, 8))
sns.heatmap(corr,cmap='RdBu', xticklabels=False, vmin=1, vmax=1, annot=True, square=True, annot_kws={'fontsize':5,'fontweight':'bold','fontfamily':'serif'})
st.markdown('#### Overall University Correlation')
st.pyplot(fig, use_container_width=True)

score = df[df['Year'] == 2024]
st.subheader('Overall Scores in 2024')
fig = plt.figure(figsize=(10,5))
sns.histplot(score['Overall Score'], label='2024', kde=True)
st.pyplot(fig,use_container_width=True)

top50 = df.nlargest(50, columns='Rank')
fig = plt.figure(figsize=(10,5))
sns.scatterplot(data=top50, x='Rank', y='Student Population')
st.markdown('#### Relationship between Student Population and University Ranking')
st.pyplot(fig,use_container_width=True)

bubble = df.loc[:100,:].sort_values(by='Rank', ascending=False)
fig = px.scatter(bubble, x='Overall Score', y='Rank', color = 'Country', width = 1000, height=600, size='Overall Score', text='Rank')
fig.update_traces(textfont_color='white')
st.subheader('Overall Scores and Ranking over Different Country')
st.plotly_chart(fig, use_container_width=True)

count = df[df['Year'] == 2022].head(50)
fig = plt.figure(figsize=(10,5))
sns.boxplot(x='Country Code', y='Overall Score', hue='Year', data=count)
st.subheader('Overall Scores in 2022')
st.pyplot(fig, use_container_width=True)

data = df.to_csv().encode('utf-8')
st.download_button(label = 'Download File', data = data, file_name = 'world_university_ranking.csv', mime = 'text/csv')


    

