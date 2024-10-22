import streamlit as st
from ml_modules import car_price_av_by

# x = st.slider('x')
# st.write(x, 'squared is', x * x)

st.markdown("""
<style>
.caption-font {font-size:30px !important;}
.title-font {font-size:18px !important;}
.write-font {font-size:14px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="caption-font">Volkswagen Polo I · Рестайлинг (2015-2020)</p>', unsafe_allow_html=True)

select_year = st.selectbox('Выберите год',[2015,2016,2017,2018,2019,2020])

# milliage_km = st.slider('Укажите пробег', min_value=10, max_value=1_000_000)
milliage_km = st.number_input('Укажите пробег')

engine_type = st.selectbox('Выберите тип двигателя',['бензин','бензин (метан)','бензин (пропан-бутан)'])

capacity = st.selectbox('Выберите объем двигателя',[1.4,1.6])

transmission_type = st.selectbox('Трансмиссия',['механика','автомат','робот'])

car_price = st.number_input('Цена в объявлении')

if st.button("Расчитать", type="primary"):
    st.markdown('<p class="title-font">Выбранные параметры:</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Год выпуска: {select_year}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Пробег: {milliage_km}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Тип двигателя: {engine_type}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Объем двигателя: {capacity}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Трансмиссия: {transmission_type}</p>', unsafe_allow_html=True)

    result = car_price_av_by.predict_price(year=select_year,milliage_km =int(milliage_km),engine_type=engine_type,capacity=capacity,transmission_type=transmission_type)

    st.markdown('<p class="title-font">Результаты расчета цены ML моделей:</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Random Forest: {int(result.get("rf")[0])} (разница: {int(car_price) - int(result.get("rf")[0])})</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">XGBoost: {int(result.get("xgboost")[0])} (разница: {int(car_price) - int(result.get("xgboost")[0])})</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="write-font">Linear Regression: {int(result.get("lr")[0])} (разница: {int(car_price) - int(result.get("lr")[0])})</p>', unsafe_allow_html=True)

    
