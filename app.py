import streamlit as st
import pandas as pd
import joblib
import statsmodels.api as sm
import numpy as np

# Carregar modelos
regression_model = joblib.load('./models/regression_model.joblib')
classifier_model = joblib.load('./models/classifier_model.joblib')


def generate_random_values(tax_class_mapping,building_main_class_mapping,sale_monthyear_mapping):
    random_data = {
        'TAX_CLASS_AT_PRESENT': [np.random.choice(list(tax_class_mapping.keys()))],
        'RESIDENTIAL_UNITS': [np.random.randint(1, 10)],
        'COMMERCIAL_UNITS': [np.random.randint(1, 5)],
        'TOTAL_UNITS': [np.random.randint(1, 15)],
        'LAND_SQUARE_FEET': [np.random.uniform(1.0, 3000.0)],
        'GROSS_SQUARE_FEET': [np.random.uniform(1.0, 1.5)],
        'BUILDING_MAIN_CLASS': [np.random.choice(list(building_main_class_mapping.keys()))],
        'DECADE_BUILT': [np.random.randint(1800, 2023)],
        'SALE_MONTHYEAR': [np.random.choice(list(sale_monthyear_mapping.keys()))]
    }
    return pd.DataFrame(random_data)


# Função para prever
def predict(data):
    # Prever valor (SALE_PRICE) usando o modelo de regressão
    sale_price_prediction = regression_model.predict(data)

    score_prediction = regression_model.score(data,sale_price_prediction)

    return sale_price_prediction[0]

def classify(data,sale_price):

    data_with_sale_price = data.copy()
    data_with_sale_price['sale_price_predict'] = sale_price

    # Classificar usando o modelo de classificação
    class_prediction = classifier_model.predict(data_with_sale_price)

    return class_prediction[0]

selected_tax_class = ''
selected_sale_monthyear = ''
selected_building_main_class = ''
residential_units = 0
commercial_units = 0
land_square_feet = 0.0
gross_square_feet = 0.0
decade_built = 0

# Converte os valores selecionados nos dropdowns para os correspondentes numéricos
tax_class_mapping = {'1': 1, '2': 2, '4': 3, '2A': 4, '2C': 5, '1B': 6, '2B': 7}
sale_monthyear_mapping = {'2022/11': 1, '2022/12': 2, '2023/01': 3, '2023/02': 4, '2023/03': 5, '2023/04': 6, '2023/05': 7, '2023/06': 8, '2023/07': 9, '2023/08': 10, '2023/09': 11, '2023/10': 12}
building_main_class_mapping = {'A': 1, 'B': 2, 'C': 3, 'R': 4, 'K': 5, 'S': 6, 'P': 7, 'V': 8, 'O': 9, 'G': 10, 'D': 11, 'F': 12, 'I': 13, 'M': 14, 'Z': 15, 'J': 16, 'E': 17, 'W': 18}


# Interface da aplicação
def main():
    st.title('Previsão do preço de venda e classificação de imóveis')
    st.markdown('*Este aplicativo busca prever de maneira razoável os valores de imóveis localizados no bairro do Queens, em Nova York e também busca tentar descobrir a qual Classe o imóvel seria classificado.*')

    # Layout em duas colunas
    col1, col2, col3 = st.columns(3)

    if 'show_col1' not in st.session_state:
        st.session_state.show_col1 = False
    if 'show_col2' not in st.session_state:
        st.session_state.show_col2 = False

    result_space = col3.empty()

    # Prever e classificar ao pressionar o botão
    if col1.button('Preencher com Valores Aleatórios'):
        st.session_state.show_col1 = True
        st.session_state.show_col2 = False

    if col2.button('Preencher Manualmente'):
        st.session_state.show_col2 = True
        st.session_state.show_col1 = False

    # Prever e classificar ao pressionar o botão
    if st.session_state.show_col1:
        random_data = generate_random_values(tax_class_mapping, building_main_class_mapping, sale_monthyear_mapping)

        # Atualize os valores nos campos de entrada
        st.session_state.selected_tax_class = col1.text_input('Grupo de imposto atualmente:', random_data['TAX_CLASS_AT_PRESENT'][0], disabled=True)
        st.session_state.selected_sale_monthyear = col1.text_input('Ano e mês da venda: (YYYY/MM)', random_data['SALE_MONTHYEAR'][0], disabled=True)
        st.session_state.selected_building_main_class = col1.text_input('Classe de construção (Matriz):', random_data['BUILDING_MAIN_CLASS'][0], disabled=True)
        st.session_state.total_units = st.session_state.residential_units + st.session_state.commercial_units
        st.session_state.residential_units = col1.number_input('Unidades Residenciais:', value=random_data['RESIDENTIAL_UNITS'][0], disabled=True)
        st.session_state.commercial_units = col1.number_input('Unidades Comerciais:', value=random_data['COMMERCIAL_UNITS'][0], disabled=True)
        st.session_state.land_square_feet = col1.number_input('Tamanho do terreno (ft²):', value=random_data['LAND_SQUARE_FEET'][0], disabled=True)
        st.session_state.gross_square_feet = col1.number_input('Tamanho bruto do terreno (ft²):', value=st.session_state.land_square_feet * np.random.uniform(1.0,1.25),disabled=True)
        st.session_state.decade_built = col1.number_input('Década de construção:', value=random_data['DECADE_BUILT'][0], disabled=True)
   
    if st.session_state.show_col2:
        # Dropdowns para os campos discretizados
        tax_class_options = ['1', '2', '4', '2A', '2C', '1B', '2B']
        st.session_state.selected_tax_class = col1.selectbox('Grupo de imposto atualmente:', tax_class_options, index=0)

        sale_monthyear_options = ['2022/11', '2022/12', '2023/01', '2023/02', '2023/03', '2023/04', '2023/05', '2023/06', '2023/07', '2023/08', '2023/09', '2023/10']
        st.session_state.selected_sale_monthyear = col1.selectbox('Ano e mês da venda: (YYYY/MM)', sale_monthyear_options, index=0)

        building_main_class_options = ['A', 'B', 'C', 'R', 'K', 'S', 'P', 'V', 'O', 'G', 'D', 'F', 'I', 'M', 'Z', 'J', 'E', 'W']
        st.session_state.selected_building_main_class = col1.selectbox('Classe de construção (Matriz):', building_main_class_options, index=0)
        

        # Outros campos de entrada
        st.session_state.residential_units = col1.number_input('Unidades Residenciais:', min_value=1, step=1)
        st.session_state.commercial_units = col1.number_input('Unidades Comerciais:', min_value=1, step=1)
        st.session_state.total_units = st.session_state.residential_units + st.session_state.commercial_units
        st.session_state.land_square_feet = col1.number_input('Tamanho do terreno (ft²):', min_value=1.0, step=1.0)
        st.session_state.gross_square_feet = col1.number_input('Tamanho bruto do terreno (ft²):', min_value=1.0, step=1.0)
        st.session_state.decade_built = col1.number_input('Década de construção:', min_value=1800, step=1)

    if col3.button('Prever e Classificar'):
        user_data = pd.DataFrame({
            'TAX_CLASS_AT_PRESENT': [tax_class_mapping[st.session_state.selected_tax_class]],
            'RESIDENTIAL_UNITS': [st.session_state.residential_units],
            'COMMERCIAL_UNITS': [st.session_state.commercial_units],
            'TOTAL_UNITS': [st.session_state.total_units],
            'LAND_SQUARE_FEET': [st.session_state.land_square_feet],
            'GROSS_SQUARE_FEET': [st.session_state.gross_square_feet],
            'BUILDING_MAIN_CLASS': [building_main_class_mapping[st.session_state.selected_building_main_class]],
            'DECADE_BUILT': [st.session_state.decade_built],
            'SALE_MONTHYEAR': [sale_monthyear_mapping[st.session_state.selected_sale_monthyear]]
        })

        sale_price_pred = predict(user_data)
        class_pred = classify(user_data, sale_price_pred)

        # Exibir resultados
        col3.subheader('Resultado da Previsão:')
        col3.write(f'Valor previsto : $ {sale_price_pred:.2f}')
        col3.subheader('Resultado da Classificação:')
        col3.write(f'Classe prevista: {class_pred}')

    col1.markdown("<style>div.Widget.row-widget.stRadio>div{flex-direction:row;}</style>", unsafe_allow_html=True)  # Adiciona espaço entre os radio buttons na coluna 1
    col1.markdown("<style>div.Widget.row-widget.stSlider>div{flex-direction:row;}</style>", unsafe_allow_html=True)  # Adiciona espaço entre os sliders na coluna 1

    space_width = 20  # Largura do espaço entre as colunas
    col1.markdown(f'<div style="margin-right:{space_width}px;"></div>', unsafe_allow_html=True)

    col2.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)

# Executar a aplicação
if __name__ == '__main__':
    main()
