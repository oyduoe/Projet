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
    esg_data = pd.read_excel("Finance verte.xlsx")
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
                **Rating ESG** : {company_data['Ratings_today']} ({company_data['Ratings_before']} → {company_data['Ratings_today']}) \n
                **Description** : {company_data['Description']} \n
                **Plan ESG** : {company_data['Plan ESG']}
                """)
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
    st.header("🎓 Éducation et Formation en Finance Durable")

    with st.expander("📚 Fondamentaux ESG"):
        st.markdown("""
        ### Les 3 Piliers ESG

        **Environnemental (E)**  
        - Émissions de GES (gaz à effet de serre), incluant les émissions directes et indirectes  
        - Efficacité énergétique dans les opérations  
        - Préservation de la biodiversité et économie circulaire  

        **Social (S)**  
        - Conditions de travail sûres et équitables  
        - Diversité et inclusion dans les effectifs  
        - Relations positives avec les communautés locales  

        **Gouvernance (G)**  
        - Indépendance et diversité des conseils d’administration  
        - Lutte contre la corruption et pratiques éthiques  
        - Droits et transparence envers les actionnaires

        ### Réglementations Clés
        - **SFDR** : règlement européen sur la transparence ESG (Articles 6, 8, 9)  
        - **Taxonomie Européenne** : classification des activités durables  
        - **Loi Pacte** : obligations ESG dans l’épargne et l’assurance-vie
        """)

    with st.expander("🏷️ Labels et Certifications"):
        st.markdown("""
        | Label       | Année | Actifs       | Particularité                       |
        |-------------|-------|--------------|-------------------------------------|
        | **ISR**     | 2016  | 800 Md€      | Analyse ESG de 90% des titres       |
        | **Greenfin**| 2015  | 36 Md€       | Exclusion des énergies fossiles     |
        | **Finansol**| 1997  | 25 Md€       | Spécialisé dans la finance solidaire

        **Différences principales :**  
        - **Finansol** : impact social direct  
        - **Greenfin** : focalisé sur l’environnement  
        - **ISR** : approche globale ESG
        """)

    with st.expander("💼 Produits Financiers Durables"):
        st.markdown("""
        **Green Bonds (Obligations Vertes)**  
        - Financement de projets environnementaux (énergies renouvelables, transports durables)  
        - Certification possible par le Climate Bonds Initiative  
        - Exemple : OAT Verte émise par la France

        **Fonds Thématiques ESG**  
        - Stratégies d’exclusion (ex : tabac, charbon)  
        - Best-in-Class : sélection des entreprises les mieux notées ESG  
        - Alignement avec les Objectifs de Développement Durable (ODD)

        **Private Equity ESG**  
        - Intégration de critères ESG dès la due diligence  
        - Suivi de l’impact environnemental et social des entreprises en portefeuille  
        - Exemples : fonds à impact, investissement dans l’agritech ou l’inclusion financière
        """)

    with st.expander("📊 Mesure de l'Impact"):
        st.markdown("""
        **Indicateurs Clés**  
        - Intensité carbone (exprimée en tCO2e/M€ investi)  
        - Pourcentage d’activités durables selon la taxonomie  
        - Score de biodiversité (MSA : Mean Species Abundance)

        **Température implicite**  
        - Évalue l’alignement d’un portefeuille avec les objectifs climatiques (ex: +1.5°C)  
        - Prend en compte les émissions directes, indirectes et de la chaîne de valeur (Scope 3)
        """)

    with st.expander("🌍 Enjeux Planétaires"):
        st.markdown("""
        **Limites Planétaires Dépassées**  
        - Changement climatique (température moyenne en hausse)  
        - Érosion de la biodiversité (extinction massive)  
        - Artificialisation des sols  
        - Perturbation des cycles biogéochimiques

        **Accord de Paris**  
        - Limiter le réchauffement en dessous de +2°C  
        - Réduction globale des émissions de GES de 43% d’ici 2030
        """)

    with st.expander("⚠️ Défis et Controverses"):
        st.markdown("""
        **Défis Sectoriels**  
        - Très faible part des fonds réellement alignés avec l’Accord de Paris  
        - Faible transparence et comparabilité des données ESG  
        - Financements persistants dans les énergies fossiles malgré les engagements

        **Greenwashing**  
        - Pratiques trompeuses d’investissement « vert »  
        - Divergences dans les notations ESG entre agences
        """)

    with st.expander("🏆 Méthodologie Goodvest (exemple d’approche stricte)"):
        st.markdown("""
        **Étapes clés :**  
        1. Exclusion totale des secteurs controversés  
        2. Analyse carbone sur l’ensemble des émissions  
        3. Mesure de l’impact sur la biodiversité  
        4. Engagement actionnarial  
        5. Validation par un comité indépendant  
        6. Transparence intégrale des choix d’investissement

        **Résultats observés :**  
        - Portefeuilles alignés avec un scénario <2°C  
        - Meilleur impact biodiversité vs fonds classiques  
        - Réduction significative des émissions financées
        """)

    with st.expander("📈 Performance vs Impact"):
        st.markdown("""
        **Comparatif ESG vs Indices Traditionnels (2014–2024)**  
        | Indice              | Performance 5 ans |
        |---------------------|------------------|
        | MSCI Europe         | +58%             |
        | MSCI Europe ESG     | +63%             |
        | MSCI PAB (Paris Al.)| +61%             |

        L’intégration ESG permet souvent de réduire la volatilité tout en conservant un rendement compétitif
        """)

    with st.expander("🧠 Quiz ESG"):
        score = 0

        st.subheader("🔰 Quiz de Base")
        q1 = st.radio("Que signifie ESG?",
                     ["Entreprise, Société, Gouvernement", 
                      "Environnement, Social, Gouvernance",
                      "Économie, Stratégie, Gestion"])
        if q1 == "Environnement, Social, Gouvernance": score += 1

        q2 = st.radio("Lequel est un critère environnemental?",
                     ["Diversité du conseil", 
                      "Émissions de GES",
                      "Politique de dividendes"])
        if q2 == "Émissions de GES": score += 1

        if st.button("✅ Vérifier Quiz de Base"):
            st.success(f"Score: {score}/2 - {'✅ Parfait!' if score == 2 else '📚 À revoir'}")

        st.subheader("📘 Quiz Intermédiaire")
        score2 = 0

        q3 = st.radio("Quel article SFDR concerne les fonds à impact ?",
                     ["Article 6", "Article 8", "Article 9"])
        if q3 == "Article 9": score2 += 1

        q4 = st.multiselect("Objectifs de la Taxonomie Européenne :", 
                           ["Adaptation au climat", "Énergie nucléaire", 
                            "Économie circulaire", "Blockchain"])
        if set(q4) == {"Adaptation au climat", "Économie circulaire"}: score2 += 1

        q5 = st.radio("Le premier green bond fut émis par :", 
                      ["World Bank (2007)", "France (2017)", "Norvège (2006)"])
        if q5 == "World Bank (2007)": score2 += 1

        q6 = st.checkbox("Les SLB lient leur coupon à des objectifs ESG")
        if q6: score2 += 1

        if st.button("📤 Soumettre Quiz Intermédiaire"):
            st.success(f"""Score: {score2}/4 - {
                '🌟 Bien joué!' if score2 >= 3 else
                '📘 Bon début, continuez' if score2 >= 2 else 
                '🔍 À retravailler'}""")

        st.subheader("📗 Quiz Expert")
        score3 = 0

        q7 = st.radio("Quel % des émissions de GES vient des énergies fossiles ?",
                     ["45%", "75%", "90%"])
        if q7 == "90%": score3 += 1

        q8 = st.multiselect("Limites planétaires déjà dépassées :", 
                           ["Cycle de l’eau", "Aérosols", "Biodiversité", "Blockchain"])
        if set(q8) == {"Aérosols", "Biodiversité"}: score3 += 1

        q9 = st.slider("Objectif de réduction des GES d’ici 2030 par rapport à 2015 :", 
                       10, 100, 43)
        if q9 == 43: score3 += 1

        if st.button("🔎 Vérifier Quiz Expert"):
            st.success(f"""Score: {score3}/3 - {
                '🏆 Maîtrise ESG!' if score3 == 3 else 
                '📗 Solide!' if score3 == 2 else 
                '🌱 Besoin de révision'}""")

def section_equipe():
    st.header("👥 Notre Équipe")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://cdn.pixabay.com/photo/2016/08/08/09/17/avatar-1577909_1280.png", use_column_width=True)
        st.markdown("**Justine Marquaille**")
    with col2:
        st.image("https://cdn.pixabay.com/photo/2016/08/08/09/17/avatar-1577909_1280.png", use_column_width=True)
        st.markdown("**Zhetian Hua**")
    with col3:
        st.image("Karine.jpg", use_column_width=True)
        st.markdown("**Karine Sun**")
    
    st.markdown("""
    ### 📂 Collaboration
    - **Méthodologie** : Le réchauffement climatique et les défis liés aux critères ESG (Environnement, Social et Gouvernance) sont des enjeux mondiaux. C’est pourquoi nous ne voulons pas simplement valoriser – et donc investir – dans les entreprises qui font déjà ce que toutes les autres devraient faire. La finance est un levier qui permet de choisir quelles entreprises nous souhaitons soutenir et quelles entreprises nous voulons voir prospérer à l’avenir.<br>
    Nous avons choisi d’aider celles qui font de réels efforts. Nous mettons l’accent sur l’effort et l’amélioration, et non uniquement sur les résultats. Il est toujours plus facile d’avoir des émissions de gaz à effet de serre quasi nulles quand on vend des gobelets réutilisables. Les choses sont bien plus complexes quand on évolue, par exemple, dans le secteur pétrolier.<br>
    C’est pourquoi nous nous concentrons sur les entreprises qui polluent beaucoup, mais qui investissent également massivement dans les initiatives ESG. Par ailleurs, l’ESG concerne avant tout la transition, et pour mettre en place une transition globale, il faut que tout le monde participe.
    - **Choix des entreprises** : Nous nous basons sur la méthodologie ESG mondiale de MSCI et analysons l’évolution de leur notation au cours des dernières années. Les notes vont de CCC à AAA. Les entreprises faisant partie de notre portefeuille sont celles qui ont connu une amélioration d’au moins deux crans entre leur note précédente et leur note actuelle.
    - **Versionning** : [GitHub Repository](https://github.com/oyduoe/Finance-durable)
    - **Description des fichiers** :
        - streamlit_app.py : Code principal de l'application Streamlit
        - requirements.txt : Fichier texte qui liste les dépendances Python nécessaires au projet
        - Data.csv : Fichier contenant les rendements de notre portefeuille
        - Finance durable.xlsx : Detail des actifs de notre portefeuille
    """, unsafe_allow_html=True)

st.title("Finance Durable")

@st.cache_data
def charger_donnees():
    return pd.read_csv("Data.csv", parse_dates=['Date'], index_col='Date').ffill().dropna(how='all')

df = charger_donnees()

page = st.sidebar.selectbox("Navigation", 
                           ["Analyse ESG",
                            "Optimisation Portefeuille", 
                            "Éducation Financière",
                            "Notre Équipe"])

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