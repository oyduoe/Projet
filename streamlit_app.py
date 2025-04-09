import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.express as px

RISK_AVERSION = 0.9
RETURN_TARGET_WEIGHT = 0.2
TAUX_SANS_RISQUE = 0.04
SEUIL_PONDERATION = 1e-4

def optimiser_portefeuille(df, mode='max_rendement'):
    nb_actifs = df.shape[1]
    matrice_cov = df.cov() * 252
    rendements_annualises = df.mean() * 252
    
    contraintes = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bornes = [(0, 1) for _ in range(nb_actifs)]
    poids_initiaux = np.ones(nb_actifs)/nb_actifs
    
    if mode == 'max_rendement':
        def objectif(w):
            return -(w @ rendements_annualises - RISK_AVERSION * np.sqrt(w.T @ matrice_cov @ w))
        
    elif mode == 'min_volatilite':
        def objectif(w):
            return (np.sqrt(w.T @ matrice_cov @ w) - RETURN_TARGET_WEIGHT * (w @ rendements_annualises))
    
    resultat = minimize(objectif, poids_initiaux, method='SLSQP', bounds=bornes, constraints=contraintes)
    
    return resultat.x

def analyser_portefeuille(rendements, poids):
    rendements_port = rendements.dot(poids)
    volatilite = rendements_port.std() * np.sqrt(252)
    rendement_annuel = rendements_port.mean() * 252
    sharpe = (rendement_annuel - TAUX_SANS_RISQUE)/volatilite if volatilite != 0 else 0
    
    df_metriques = pd.DataFrame({
        'Rendement Annuel (%)': [rendement_annuel * 100],
        'Volatilité Annuelle (%)': [volatilite * 100],
        'Ratio de Sharpe': [sharpe]
    }).round(2)
    
    df_actifs = pd.DataFrame({
        'Actif': rendements.columns[np.where(poids > SEUIL_PONDERATION)[0]],
        'Pondération (%)': [f"{p * 100:.2f}%" for p in poids[poids > SEUIL_PONDERATION]]
    })
    
    return df_metriques, df_actifs

def plot_perf(rendements, poids, titre):
    fig, ax = plt.subplots()
    (1 + rendements.dot(poids)).cumprod().plot(title=titre, ax=ax)
    plt.grid(True)
    return fig

def section_esg():
    st.header("🌍 Analyse ESG Détaillée")
    esg_data = pd.read_excel(r"/workspaces/Projet/Finance verte.xlsx")
    
    logo_map = {
        'AB Sagax': r"C:\Users\Popcorn\Downloads\SAGAX_logo_gra-1.jpg",
        'ADMIRAL GROUP PLC': r"C:\Users\Popcorn\Downloads\Admiral.jpg",
        # Ajouter toutes les entreprises ici...
    }
    
    rating_evolution_map = {
        'AB Sagax': r"D:\HONOR Share\Screenshot\AB Sagax.bmp",
        'ADMIRAL GROUP PLC': r"chemin_vers_image_evolution_admiral.bmp",
        # Ajouter toutes les entreprises ici...
    }

    rating_order = ['CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']
    rating_colors = {
        'CCC': '#FF6B6B',
        'B': '#FF9F9F',
        'BB': '#FFD3D3',
        'BBB': '#E8F5E9',
        'A': '#C8E6C9',
        'AA': '#A5D6A7',
        'AAA': '#66BB6A'
    }
    
    esg_data['Logo'] = esg_data['Companies'].apply(lambda x: logo_map.get(x, r"C:\Users\Popcorn\Downloads\placeholder.png"))
    esg_data['Rating_Evolution_Image'] = esg_data['Companies'].apply(lambda x: rating_evolution_map.get(x, r"C:\Users\Popcorn\Downloads\placeholder_evolution.png"))
    esg_data['Rating'] = pd.Categorical(esg_data['Ratings_today'], categories=rating_order, ordered=True)
    
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        selected_countries = st.multiselect(
            "Choisir des pays",
            options=esg_data['Country'].unique(),
            default=[],
            key='esg_country'
        )
    with col2:
        selected_sectors = st.multiselect(
            "Choisir des secteurs",
            options=esg_data['Industry'].unique(),
            default=[],
            key='esg_sector'
        )
    with col3:
        selected_company = st.selectbox(
            "Rechercher un actif",
            options=[''] + sorted(esg_data['Companies'].unique()),
            format_func=lambda x: "Tous les actifs" if x == '' else x,
            key='esg_search'
        )
    
    filtered_data = esg_data.copy()
    if selected_countries:
        filtered_data = filtered_data[filtered_data['Country'].isin(selected_countries)]
    if selected_sectors:
        filtered_data = filtered_data[filtered_data['Industry'].isin(selected_sectors)]
    if selected_company:
        filtered_data = filtered_data[filtered_data['Companies'] == selected_company]
    
    if not filtered_data.empty:
        if selected_company:
            company_data = filtered_data.iloc[0]
            st.subheader(f"Fiche Entreprise : {company_data['Companies']}")
            col_info, col_logo = st.columns([3, 1])
            with col_info:
                st.markdown(f"""
                **Pays** : {company_data['Country']} \n
                **Secteur** : {company_data['Industry']} \n
                **Rating ESG Actuel** : {company_data['Rating']} \n
                **Description** : A faire
                """)
                st.image(company_data['Rating_Evolution_Image'], 
                        caption="Historique des ratings ESG",
                        width=400)
            with col_logo:
                st.image(company_data['Logo'], 
                        caption=company_data['Companies'],
                        width=150)
            st.divider()
        
        with st.expander("Carte géographique des actifs"):
            country_counts = filtered_data['Country'].value_counts().reset_index()
            fig = px.choropleth(
                country_counts,
                locations='Country',
                locationmode='country names',
                color='count',
                hover_name='Country',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"{len(filtered_data)} actifs trouvés")
        col_table, col_metrics = st.columns([3, 1])
        with col_table:
            def color_ratings(val):
                return f'background-color: {rating_colors.get(val, "white")}'
            st.dataframe(
                filtered_data[['Companies', 'Country', 'Industry', 'Rating']]
                .sort_values('Rating', ascending=False)
                .reset_index(drop=True)
                .style.map(color_ratings, subset=['Rating']),
                height=500
            )
        
        with col_metrics:
            st.metric("Meilleur rating", filtered_data['Rating'].max())
            st.metric("Pays le plus représenté", filtered_data['Country'].mode()[0])
            st.metric("Secteur dominant", filtered_data['Industry'].mode()[0])
    else:
        st.warning("Aucun actif ne correspond aux critères sélectionnés")

def section_education():
    st.header("🎓 Éducation Financière")
    
    with st.expander("📚 Guide des Critères ESG"):
        st.markdown("""
        **Environnemental (E)**
        - Émissions CO2
        - Utilisation des ressources
        - Biodiversité
        
        **Social (S)**
        - Conditions travail
        - Diversité
        - Relations communautés
        
        **Gouvernance (G)**
        - Éthique des affaires
        - Rémunération dirigeants
        - Droits actionnaires
        """)
    
    with st.expander("🧠 Quiz ESG"):
        score = 0
        q1 = st.radio("Que signifie ESG?",
                     ["Entreprise, Société, Gouvernement", 
                      "Environnement, Social, Gouvernance",
                      "Économie, Stratégie, Gestion"])
        if q1 == "Environnement, Social, Gouvernance":
            score += 1
        q2 = st.radio("Lequel est un critère environnemental?",
                     ["Diversité du conseil", 
                      "Émissions de GES",
                      "Politique dividendes"])
        if q2 == "Émissions de GES":
            score += 1
        if st.button("Vérifier les réponses"):
            st.success(f"Score: {score}/2 - {'✅ Parfait!' if score ==2 else '📚 À revoir'}")

def section_equipe():
    st.header("👥 Notre Équipe")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://cdn.pixabay.com/photo/2016/08/08/09/17/avatar-1577909_1280.png", width=150)
        st.markdown("**Justine Marquaille**")
    with col2:
        st.image("https://cdn.pixabay.com/photo/2016/08/08/09/17/avatar-1577909_1280.png", width=150)
        st.markdown("**Zhetian Hua**")
    with col3:
        st.image("https://cdn.pixabay.com/photo/2016/08/08/09/17/avatar-1577909_1280.png", width=150)
        st.markdown("**Karine Sun**")
    
    st.markdown("""
    ### 📂 Collaboration
    - **Méthodologie** : Agile
    - **Versionning** : [GitHub Repository](https://github.com/oyduoe/Finance-durable)
    - **Description des fichiers** :
        - app.py : Code principal de l'application Streamlit
        - Data.csv : Fichier contenant les rendements de notre portefeuille
        - Finance durable.xlsx : Detail des actifs de notre portefeuille
    """)

st.title("Finance Durable")

@st.cache_data
def charger_donnees():
    return pd.read_csv(r"/workspaces/Projet/Data.csv", parse_dates=['Date'], index_col='Date').ffill().dropna(how='all')

df = charger_donnees()

# Menu de navigation
page = st.sidebar.selectbox("Navigation", 
                           ["Analyse ESG",
                            "Optimisation Portefeuille", 
                            "Éducation Financière",
                            "Notre Équipe"])

# Gestion des pages
if page == "Optimisation Portefeuille":
    st.header("📈 Optimisation Portefeuille")
    st.write("Quel est votre objectif d'investissement ?")

    col1, col2, col3 = st.columns(3)
    with col1:
        btn_max_ret = st.button('Maximiser le rendement')
    with col2:
        btn_min_vol = st.button('Minimiser la volatilité')
    with col3:
        btn_esg = st.button('ESG (à venir)')

    if btn_max_ret:
        poids = optimiser_portefeuille(df, 'max_rendement')
        metriques, actifs = analyser_portefeuille(df, poids)
        
        st.subheader("Portefeuille - Maximisation du rendement")
        st.dataframe(metriques)
        st.dataframe(actifs)
        st.pyplot(plot_perf(df, poids, 'Performance - Rendement Maximisé'))

    if btn_min_vol:
        poids = optimiser_portefeuille(df, 'min_volatilite')
        metriques, actifs = analyser_portefeuille(df, poids)
        
        st.subheader("Portefeuille - Minimisation de la volatilité")
        st.dataframe(metriques)
        st.dataframe(actifs)
        st.pyplot(plot_perf(df, poids, 'Performance - Volatilité Minimisée'))

    if btn_esg:
        st.write("Fonctionnalité ESG en cours de développement...")
    
elif page == "Analyse ESG":
    section_esg()
    
elif page == "Éducation Financière":
    section_education()
    
elif page == "Notre Équipe":
    section_equipe()