import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Конфигурация страницы
st.set_page_config(
    page_title="Прогноз недвижимости",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Загрузка артефакта (кэшируется)
@st.cache_resource
def load_artifact():
    try:
        artifact = joblib.load('full_pipeline.joblib')
        return artifact
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


@st.cache_data
def load_dataset():
    for path in ['real_estate_data.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            if 'city' not in df.columns and 'address' in df.columns:
                df['city'] = df['address'].str.split('/').str[0]
            if 'price_currency' in df.columns:
                df = df[df['price_currency'] == 'TRY']
            df = df[(df['price'] > 50_000) & (df['price'] < 5_000_000)]
            df = df.dropna(subset=['price', 'size', 'city'])
            return df
    return None


artifact = load_artifact()

# Sidebar — навигация
st.sidebar.title("Меню")
page = st.sidebar.radio(
    "Выберите страницу:",
    ["Прогноз", "Дашборд", "Справка"],
)

st.sidebar.markdown("---")
if artifact:
    st.sidebar.success(f"Модель: {artifact['best_model_name']}")
    st.sidebar.metric("R²", f"{artifact['metrics']['R²']:.3f}")
    st.sidebar.metric("MAE", f"{artifact['metrics']['MAE']:,.0f} TRY")
else:
    st.sidebar.error("Модель не загружена")

st.sidebar.markdown("---")
st.sidebar.caption(f"Версия 2.0.0 · {datetime.now().year}")


# Страница 1: ПРОГНОЗ
if page == "Прогноз":
    st.title("Прогноз стоимости недвижимости")
    st.markdown("Заполните характеристики объекта — модель рассчитает цену.")

    if not artifact:
        st.error("Модель не загружена. Проверьте `full_pipeline.joblib`.")
        st.stop()

    meta = artifact['metadata']

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            size = st.number_input("Площадь (м²)", min_value=20, max_value=1000, value=120, step=5)
            rooms_num = st.number_input("Комнат", min_value=0, max_value=10, value=3)
            building_age = st.number_input("Возраст здания (лет)", min_value=0, max_value=100, value=5)
            total_floors = st.number_input("Всего этажей", min_value=1, max_value=50, value=10)
            sub_type = st.selectbox("Тип объекта", meta['sub_types'], index=meta['sub_types'].index('Daire'))

        with col2:
            city = st.selectbox("Город", meta['cities'], index=meta['cities'].index('İstanbul') if 'İstanbul' in meta['cities'] else 0)
            floor_category = st.selectbox("Категория этажа", meta['floor_categories'], index=meta['floor_categories'].index('floor_3') if 'floor_3' in meta['floor_categories'] else 0)
            heating = st.selectbox("Отопление", meta['heating_types'], index=meta['heating_types'].index('Kombi (Doğalgaz)') if 'Kombi (Doğalgaz)' in meta['heating_types'] else 0)
            listing = st.selectbox("Тип предложения", [1, 2], format_func=lambda x: "Продажа" if x == 1 else "Аренда")

        submitted = st.form_submit_button("Рассчитать стоимость", use_container_width=True)

    if submitted:
        with st.spinner("Расчёт..."):
            input_data = {
                'size': float(size),
                'rooms_num': int(rooms_num),
                'building_age_num': float(building_age),
                'total_floor_count_num': float(total_floors),
                'sub_type': sub_type,
                'listing_type': int(listing),
                'heating_type': heating,
                'city': city,
                'floor_category': floor_category,
            }

            row = pd.DataFrame([input_data])
            X_enc = artifact['preprocessor'].transform(row)
            log_pred = artifact['model'].predict(X_enc)[0]
            price = float(np.expm1(log_pred))
            mae = artifact['metrics']['MAE']

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Прогноз", f"{price:,.0f} ₺")
        col_b.metric("Минимум (−MAE)", f"{max(0, price-mae):,.0f} ₺")
        col_c.metric("Максимум (+MAE)", f"{price+mae:,.0f} ₺")

        st.info(f"**Диапазон:** {max(0, price-mae):,.0f} — {price+mae:,.0f} ₺  ·  **Точность:** ±{mae:,.0f} TRY")

        with st.expander("Показать входные данные"):
            st.json(input_data)


# Страница 2: ДАШБОРД
elif page == "Дашборд":
    st.title("Дашборд проекта")

    df = load_dataset()

    # --- Метрики модели ---
    if artifact:
        m = artifact['metrics']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{m['R²']:.3f}")
        c2.metric("MAE", f"{m['MAE']:,.0f} TRY")
        c3.metric("MAPE", f"{m['MAPE']:.1f}%")
        c4.metric("Модель", artifact['best_model_name'])

    if df is None:
        st.warning("Датасет не найден. Положите `real_estate_data.csv` рядом с `app.py`.")
        st.stop()

    # --- Summary ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", f"{len(df):,}")
    c2.metric("Средняя цена", f"{df['price'].mean():,.0f} ₺")
    c3.metric("Медианная цена", f"{df['price'].median():,.0f} ₺")
    c4.metric("Средняя площадь", f"{df['size'].mean():.1f} м²")

    st.markdown("---")

    # --- График городов ---
    st.subheader("Средние цены по топ-10 городам")
    city_stats = (
        df.groupby('city')['price']
          .agg(['mean', 'count'])
          .sort_values('count', ascending=False)
          .head(10)
          .reset_index()
    )
    st.bar_chart(city_stats.set_index('city')['mean'])

    # --- Распределения ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение цен")
        price_bins = pd.cut(df['price'], bins=20).value_counts().sort_index()
        price_bins.index = [f"{int(i.left/1000)}k–{int(i.right/1000)}k" for i in price_bins.index]
        st.bar_chart(price_bins)

    with col2:
        st.subheader("Распределение площади")
        size_data = df[(df['size'] > 0) & (df['size'] < 500)]
        size_bins = pd.cut(size_data['size'], bins=15).value_counts().sort_index()
        size_bins.index = [f"{int(i.left)}–{int(i.right)}" for i in size_bins.index]
        st.bar_chart(size_bins)

    # --- Типы объектов ---
    if 'sub_type' in df.columns:
        st.subheader("Типы объектов")
        subtype_counts = df['sub_type'].value_counts().head(8)
        st.bar_chart(subtype_counts)


# Страница 3: СПРАВКА
elif page == "Справка":
    st.title("Справка")

    st.header("О приложении")
    st.markdown(
        "Веб-приложение для прогнозирования стоимости недвижимости в Турции. "
        "Использует ML-модель **RandomForestRegressor**, обученную на датасете "
        "из ~166 тыс. объектов."
    )

    st.header("Как пользоваться")
    st.markdown("""
    1. Перейдите на страницу **Прогноз**
    2. Заполните характеристики объекта
    3. Нажмите **«Рассчитать стоимость»**
    4. Получите прогноз цены и интервал ±MAE
    """)

    st.header("О модели")
    st.markdown("""
    - **Алгоритм:** RandomForestRegressor (200 деревьев, max_depth=25)
    - **Целевая переменная:** log₁ₚ(price_try)
    - **Признаки:** size, rooms_num, building_age_num, total_floor_count_num,
      sub_type, listing_type, heating_type, city, floor_category
    """)

    if artifact:
        m = artifact['metrics']
        st.subheader("Метрики на тестовой выборке")
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{m['R²']:.3f}")
        c2.metric("MAE", f"{m['MAE']:,.0f} TRY")
        c3.metric("MAPE", f"{m['MAPE']:.1f}%")

    st.header("Ограничения модели")
    st.markdown("""
    - Обучена на диапазоне цен **85,000 – 3,500,000 TRY** (winsorization 1%-99%)
    - Менее точна для элитного сегмента (>2M TRY)
    - Не учитывает состояние ремонта, вид из окна, срочность продажи
    - Прогноз — ориентировочный, не заменяет оценку риэлтора
    """)

    st.header("Технологии")
    st.markdown("""
    - **Backend:** Streamlit
    - **ML:** scikit-learn 1.6.1
    - **Frontend:** Streamlit widgets + Charts
    - **Данные:** ~166k объектов недвижимости Турции
    """)

    st.markdown("---")
    st.caption(f"Версия 2.0.0 · Автор: Юсупов Айдар Ришатович 23П-1 · {datetime.now().year}")