import streamlit as st
import time

st.set_page_config(
    page_title="Smart Sale Fortaleza",
    page_icon="./assets/sale_icon_264139.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CSS
st.markdown("""
    <style>
        /* Fundo geral */
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1e293b;
        }

        /* Título da sidebar */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #22c55e;
            font-weight: 700;
        }

        /* Textos e inputs */
        .stTextInput > div > div > input {
            background-color: #1f2937;
            color: #f9fafb;
            border-radius: 8px;
            border: 1px solid #374151;
        }

        /* Botões */
        .stButton>button {
            background-color: #22c55e !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            height: 30px !important; /* força altura igual ao input */
            width: 100% !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: 0.2s;
            margin-top: 2px !important; /* corrige alinhamento */
        }

        .stButton>button:hover {
            background-color: #16a34a !important;
            transform: scale(1.03);
        }

        .stButton>button:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Cabeçalho */
        .main-title {
            color: #22c55e;
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: 10px;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        /* Placeholder */
        .stAlert {
            background-color: #1f2937;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)




#sidebar
with st.sidebar:
    st.header("Filtros")
    classe = st.multiselect("Classe Social", ["A", "B", "C"])
    tipo = st.selectbox("Tipo Comercial", ["Mercado", "Farmácia", "Academia"])
    bairro = st.multiselect("Bairro", ["Centro", "Aldeota", "Meireles", "Varjota", "Montese"])
    if st.button("Aplicar Filtros"):
        st.success("Filtros aplicados com sucesso!")



st.markdown("<h1 class='main-title'>Smart Sale Fortaleza</h1>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Encontre os melhores locais para vender seu produto em Fortaleza</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 6, 2])
with col2:
    # Campo + botão estilo ChatGPT
    col_input, col_button = st.columns([7, 1])

    with col_input:
        produto = st.text_input(
            "",
            placeholder="Digite o produto que deseja vender...",
            label_visibility="collapsed",
            key="produto_input"
        )

    with col_button:
        enviar = st.button("Send", key="enviar_button")

    # Estilização
    st.markdown("""
    <style>

        div[data-baseweb="input"] > div:focus-within {
            border: 1px solid #22c55e !important;
        }

        .stButton>button {
            background-color: #22c55e !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.7rem 0.9rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            transition: 0.2s;
        }
        .stButton>button:hover {
            background-color: #16a34a !important;
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

    # Placeholder de feedback
    placeholder = st.empty()
    if enviar:
        if not produto.strip():
            placeholder.warning("Você não digitou nada ainda!")
            time.sleep(2)
            placeholder.empty()
        else:
            placeholder.success(f"Produto digitado: **{produto}**")
            time.sleep(2)
            placeholder.empty()