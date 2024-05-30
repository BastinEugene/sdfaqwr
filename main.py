import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, \
    GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import io


def user_input():
    data_source = st.selectbox("Выберите источник данных", ["Excel", "CSV"], index=0, key="data_source_selectbox")
    if data_source == "CSV":
        file = st.file_uploader("Загрузите CSV файл", type="csv", key="csv_file_uploader")
        return file, "CSV" if file else (None, None)
    elif data_source == "Excel":
        file = st.file_uploader("Загрузите Excel файл", type="xlsx", key="excel_file_uploader")
        return file, "Excel" if file else (None, None)
    return None, None

def load_data(source, source_type):
    try:
        if source_type == "CSV":
            data = pd.read_csv(source)
        elif source_type == "Excel":
            data = pd.read_excel(source)
        return data
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None


def standardize_data(data):
    try:
        data.columns = [col.lower() for col in data.columns]
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if 'unnecessary_column' in data.columns:
            data = data.drop(columns=['unnecessary_column'])
        return data
    except Exception as e:
        st.error(f"Ошибка при стандартизации данных: {e}")
        return data


def fill_missing_values(data):
    try:
        def fill_missing_with_arima(column):
            if column.dropna().shape[0] > 0:  # Проверяем, что есть непустые значения
                model = ARIMA(column.dropna(), order=(1, 1, 1))
                model_fit = model.fit()
                start_index = len(column.dropna())
                end_index = len(column) - 1
                if start_index <= end_index:
                    filled = model_fit.predict(start=start_index, end=end_index, dynamic=True)
                    column.loc[column.isna()] = filled
            return column

        def fill_missing_with_ml(column):
            y = column
            X = np.arange(len(column)).reshape(-1, 1)
            if y.dropna().shape[0] > 0:  # Проверяем, что есть непустые значения
                reg = GradientBoostingRegressor()
                reg.fit(X[~y.isna()], y.dropna())
                y_pred = reg.predict(X[y.isna()])
                column.loc[column.isna()] = y_pred
            return column

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        cat_cols = data.select_dtypes(include=['object']).columns

        # Заполнение пропусков в числовых столбцах
        for column in numeric_cols:
            if data[column].isna().sum() > 0:
                arima_filled = fill_missing_with_arima(data[column])
                ml_filled = fill_missing_with_ml(data[column])
                data.loc[data[column].isna(), column] = 0.5 * arima_filled[data[column].isna()] + 0.5 * ml_filled

        # Заполнение пропусков в категориальных столбцах
        imputer = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = imputer.fit_transform(data[cat_cols])

        return data
    except Exception as e:
        st.error(f"Ошибка при заполнении пропусков: {e}")
        return data


def normalize_data(data):
    try:
        def adaptive_normalization(column):
            if column.skew() > 1 or column.skew() < -1:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            return scaler.fit_transform(column.values.reshape(-1, 1))

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column != 'loan_id':  # Исключаем 'loan_id' из нормализации
                data[column] = adaptive_normalization(data[column])
        return data
    except Exception as e:
        st.error(f"Ошибка при нормализации данных: {e}")
        return data


def clean_data(data):
    try:
        # Выделение числовых данных для модели
        X = data.select_dtypes(include=[np.number]).copy()

        # Определение целевого признака (анализируем все числовые столбцы)
        target_columns = X.columns

        # Обучение моделей для каждого числового столбца
        for target in target_columns:
            y = X[target]
            X_train = X.drop(columns=[target])
            reg1 = RandomForestRegressor(n_estimators=50, random_state=1)
            reg2 = GradientBoostingRegressor(n_estimators=50, random_state=1)
            ensemble_reg = VotingRegressor(estimators=[('rf', reg1), ('gb', reg2)])

            # Обучение ансамблевой модели
            ensemble_reg.fit(X_train, y)

            # Предсказание и определение аномалий
            preds = ensemble_reg.predict(X_train)

            # Вычисление отклонений и метки аномалии
            errors = np.abs(preds - y)
            threshold = np.mean(errors) + 3 * np.std(errors)
            data['anomaly'] = np.where(errors > threshold, -1, 1)

            # Очистка данных
            data_clean = data[data['anomaly'] != -1].copy()
            data_clean = data_clean.drop(columns=['anomaly'])

        # Удаление строк с пропусками
        data_clean = data_clean.dropna().reset_index(drop=True)

        return data_clean
    except Exception as e:
        print(f"Ошибка при очистке данных: {e}")
        return data

def encode_categorical_data(data):
    try:
        cat_cols = data.select_dtypes(include=['object', 'category']).columns

        # Преобразуем все значения в категориальных столбцах в строки и заполняем пропуски
        for col in cat_cols:
            data[col] = data[col].astype(str).fillna('missing')

        for col in cat_cols:
            if data[col].nunique() == 2:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            else:
                ohe = OneHotEncoder(drop='first', sparse_output=False)
                encoded = ohe.fit_transform(data[[col]])
                encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]))

                data = pd.concat([data.drop(columns=[col]), encoded_df], axis=1)

        # Дополнительная проверка на наличие '_nan'
        columns_to_drop = [col for col in data.columns if '_nan' in col]
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)

        return data
    except Exception as e:
        st.error(f"Ошибка при обработке категориальных данных: {e}")
        return data

def export_data(data, step_name):
    export_format = st.selectbox(f"Выберите формат экспорта для {step_name}", ["Excel", "CSV"], index=0, key=f"export_format_{step_name}")
    if export_format == "CSV":
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label=f"Скачать {step_name} CSV", data=csv, file_name=f"{step_name}_data.csv", mime='text/csv')
    elif export_format == "Excel":
        towrite = io.BytesIO()
        data.to_excel(towrite, index=False, engine='xlsxwriter')
        towrite.seek(0)
        st.download_button(label=f"Скачать {step_name} Excel", data=towrite, file_name=f"{step_name}_data.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


def main():
    st.title("Система предобработки данных")

    source, source_type = user_input()
    if source and source_type:
        data = load_data(source, source_type)
        if data is not None:
            st.write("Данные загружены успешно. Нажмите кнопку ниже, чтобы начать обработку данных.")
            if st.button("Начать обработку данных"):
                data = standardize_data(data)
                with st.expander("Данные после стандартизации"):
                    st.write(data.tail(4))
                    export_data(data, "standardized")

                data = fill_missing_values(data)
                with st.expander("Данные после заполнения пропусков"):
                    st.write(data.tail(4))
                    export_data(data, "filled_missing")

                data = clean_data(data)
                with st.expander("Данные после очистки"):
                    st.write(data.tail(4))
                    export_data(data, "cleaned")

                data = normalize_data(data)
                with st.expander("Данные после нормализации"):
                    st.write(data.tail(4))
                    export_data(data, "normalized")

                data = encode_categorical_data(data)
                with st.expander("Данные после кодирования категориальных данных"):
                    st.write(data.tail(4))
                    export_data(data, "encoded")

                # Финальная кнопка для скачивания предобработанных данных
                st.write("Обработка данных завершена. Вы можете скачать предобработанные данные ниже.")
                export_format = st.selectbox("Выберите формат для скачивания", ["Excel", "CSV"], index=0, key="final_export_format")
                if export_format == "CSV":
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Скачать предобработанные данные CSV", data=csv, file_name="preprocessed_data.csv", mime='text/csv')
                elif export_format == "Excel":
                    towrite = io.BytesIO()
                    data.to_excel(towrite, index=False, engine='xlsxwriter')
                    towrite.seek(0)
                    st.download_button(label="Скачать предобработанные данные Excel", data=towrite, file_name="preprocessed_data.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    main()