import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.express as px

RISK_AVERSION = 0.9 # Contrôle le compromis rendement/risque (0 = pur rendement, 1 = plus prudent)
RETURN_TARGET_WEIGHT = 0.2 # Poids du rendement dans l'optimisation de volatilité
TAUX_SANS_RISQUE = 0.04 # Taux pour le ratio de sharpe
SEUIL_PONDERATION = 1e-4 # 0.01% de poids minimum pour afficher un actif
ALPHA = 0.2 # Rendement
GAMMA = 0.9 # Notes subjectives

ESG_dict = {
    "EQT": 2,
    "SAGA-B.ST": 2,
    "ACGBY": 2,
    "ATEYY": 2,
    "2395.TW": 2,
    "ADM.L": 3,
    "AFL": 2,
    "ANET": 2,
    "ARES": 2,
    "ACGL": 2,
    "300999.SZ": 4,
    "AU": 3,
    "AIR.PA": 2,
    "2618.TW": 2,
    "AON": 2,
    "A17U.SI": 2,
    "TEMN.SW": 2,
    "HOLN.SW": 2,
    "2802.T": 2,
    "MB.MI": 4,
    "BAMI.MI": 2,
    "PST.MI": 2,
    "ALE.WA": 4,
    "AVB": 2,
    "AVY": 3,
    "2588.HK": 2,
    "INDIGO.NS": 2,
    "9202.T": 2,
    "KMI": 3
}

def optimiser_portefeuille(df, mode='max_rendement', notes=None, max_poids_par_actif=0.25):
    nb_actifs = df.shape[1]
    matrice_cov = df.cov() * 252
    rendements_annualises = df.mean() * 252
    
    contraintes = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bornes = [(0, max_poids_par_actif) for _ in range(nb_actifs)]
    poids_initiaux = np.ones(nb_actifs) / nb_actifs
    
    if mode == 'max_rendement': # Objectif : Max(rendement) - λ*Risque
        def objectif(w):
            return -(w @ rendements_annualises - RISK_AVERSION * np.sqrt(w.T @ matrice_cov @ w))
        
    elif mode == 'min_volatilite': # Objectif : Min(volatilité) - γ*Rendement
        def objectif(w):
            return (np.sqrt(w.T @ matrice_cov @ w) - RETURN_TARGET_WEIGHT * (w @ rendements_annualises))
    
    elif mode == 'esg' and notes is not None:
        notes_vecteur = np.array([notes[col] for col in df.columns])
        def objectif(w):
            rendement = w @ rendements_annualises
            risque = np.sqrt(w.T @ matrice_cov @ w)
            score = w @ notes_vecteur
            return -(ALPHA * rendement - RISK_AVERSION * risque + GAMMA * score)
    
    elif mode == 'egale_pondere':
        return poids_initiaux
    
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
        st.image("https://api.dicebear.com/7.x/initials/svg?seed=Justine+Marquaille", use_column_width=True)
        st.markdown("**Justine Marquaille**")
    with col2:
        st.image("https://api.dicebear.com/7.x/initials/svg?seed=Zhetian+Hua", use_column_width=True)
        st.markdown("**Zhetian Hua**")
    with col3:
        st.image("https://api.dicebear.com/7.x/initials/svg?seed=Karine+Sun", use_column_width=True)
        st.markdown("**Karine Sun**")
    
    st.markdown("""
    ### 📂 Collaboration
    - **Méthodologie** : Le réchauffement climatique et les défis ESG sont un problème mondial. C'est pourquoi nous ne voulons pas nous contenter de valoriser et donc d'investir dans des entreprises qui font déjà ce que toutes les autres devraient faire. La finance est un moyen de choisir les entreprises que nous voulons aider et celles que nous voulons voir dans le futur. Nous avons choisi d'aider celles qui font de grands efforts. Nous nous concentrons sur l'effort et l'amélioration, et non sur le résultat. On a appelé cette méthodologie Best-Effort. Il est toujours facile d'avoir des émissions de gaz proches de zéro lorsque vous vendez des gobelets réutilisables. Les choses sont beaucoup plus difficiles lorsque vous travaillez dans le secteur pétrolier, par exemple. C'est pourquoi nous nous concentrons sur les entreprises qui polluent beaucoup mais qui investissent beaucoup dans les questions ESG. En outre, l'ESG est une question de transition, et pour mettre en œuvre une transition mondiale, nous avons besoin de l'implication de tous.
    - **Choix des entreprises** : Nous nous appuyons sur la méthodologie ESG globale de MSCI et analysons l'amélioration de la notation au cours des dernières années. Les notes vont de CCC à AAA. Les entreprises qui font partie de notre univers d’investissement sont celles qui se sont améliorées d'au moins deux niveaux entre la notation précédente et celle d'aujourd'hui. Une fois cet univers filtré, les analyses de prix et rendements sont effectuées sur la période allant du 1er janvier 2023 au 31 décembre 2024. Un de nos portefeuille (le troisième) a d’ailleurs été construit pour investir dans les entreprises qui avaient produit le plus d’efforts, c’est-à-dire qui ont fourni un gros effort et donc ont vu une nette amélioration dans leur notation.
    - **Versionning** : [GitHub Repository](https://github.com/oyduoe/Projet)
    - **Description des fichiers** :
        - Finance verte.xlsx : Fichier détaillant les actifs du portefeuille, incluant notamment des informations sur leur classification selon des critères de finance durable
        - Devise.csv : Fichier contenant les taux de change des différentes devises utilisées dans le portefeuille. Il permet de convertir les prix des actifs dans une devise commune : l'Euro
        - Data.csv : Fichier contenant les rendements historiques des actifs composant le portefeuille. Il est utilisé pour analyser la performance d'un portefeuille
        - data.py : Script Python qui récupère les prix des actifs via l’API Yahoo Finance, convertit les prix en euros à l’aide du fichier Devise.csv, calcule les rendements et génère Data.csv
        - streamlit_app.py : Code principal de l'application Streamlit
        - requirements.txt : Fichier texte listant toutes les dépendances Python nécessaires pour faire tourner l’application
    """, unsafe_allow_html=True)

st.title("Finance Durable")

@st.cache_data
def charger_donnees():
    return pd.read_csv("Data.csv", index_col=0, parse_dates=True)

df = charger_donnees()

page = st.sidebar.selectbox("Navigation", 
                           ["Analyse ESG",
                            "Optimisation Portefeuille", 
                            "Éducation Financière",
                            "Notre Équipe"])

if page == "Optimisation Portefeuille":
    st.header("📈 Optimisation Portefeuille")
    st.write("Quel est votre objectif d'investissement ?")

    st.markdown("""
    <style>
        div[data-testid="column"] {
            gap: 0.5rem;
        }
        .stButton > button {
            width: 100%;
            white-space: nowrap;
        }
    </style>
    """, unsafe_allow_html=True)

    if 'mode_selectionne' not in st.session_state:
        st.session_state.mode_selectionne = None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Maximiser rendement'):
            st.session_state.mode_selectionne = 'max_rendement'
    with col2:
        if st.button('Minimiser volatilité'):
            st.session_state.mode_selectionne = 'min_volatilite'
    with col3:
        if st.button('ESG'):
            st.session_state.mode_selectionne = 'esg'
    with col4:
        if st.button('Personnalisable'):
            st.session_state.mode_selectionne = 'egale_pondere'

    mode = st.session_state.mode_selectionne

    if mode == 'max_rendement':
        poids = optimiser_portefeuille(df, 'max_rendement')
        metriques, actifs = analyser_portefeuille(df, poids)
        st.subheader("📈 Portefeuille - Maximisation du rendement")
        st.dataframe(metriques)
        st.dataframe(actifs)
        st.pyplot(plot_perf(df, poids, 'Performance - Rendement Maximisé'))

    elif mode == 'min_volatilite':
        poids = optimiser_portefeuille(df, 'min_volatilite')
        metriques, actifs = analyser_portefeuille(df, poids)
        st.subheader("📉 Portefeuille - Minimisation de la volatilité")
        st.dataframe(metriques)
        st.dataframe(actifs)
        st.pyplot(plot_perf(df, poids, 'Performance - Volatilité Minimisée'))

    elif mode == 'esg':
        poids = optimiser_portefeuille(df, 'esg', ESG_dict)
        metriques, actifs = analyser_portefeuille(df, poids)
        st.subheader("🌱 Portefeuille - ESG")
        st.dataframe(metriques)
        st.dataframe(actifs)
        st.pyplot(plot_perf(df, poids, 'Performance - ESG'))

    elif mode == 'egale_pondere':
        st.subheader("Sélectionnez les entreprises pour votre portefeuille personnalisé")
        esg_data = pd.read_excel("Finance verte.xlsx")
        esg_filtered = esg_data[esg_data['Tickers'].isin(df.columns)]
        noms_disponibles = sorted(esg_filtered['Companies'].unique())
        entreprises_choisies = st.multiselect("Choisissez les actifs :", options=noms_disponibles)
        tickers_choisis = esg_filtered[esg_filtered['Companies'].isin(entreprises_choisies)]['Tickers'].tolist()
        if tickers_choisis:
            df_selection = df[tickers_choisis].dropna()
            poids = optimiser_portefeuille(df_selection, 'egale_pondere')
            metriques, actifs = analyser_portefeuille(df_selection, poids)
            st.subheader("⚖️ Portefeuille - Pondération Égale Personnalisée")
            st.dataframe(metriques)
            st.dataframe(actifs)
            st.pyplot(plot_perf(df_selection, poids, 'Performance - Personnalisé'))
        else:
            st.info("Veuillez sélectionner au moins un actif.")

elif page == "Analyse ESG":
    section_esg()
    
elif page == "Éducation Financière":
    section_education()
    
elif page == "Notre Équipe":
    section_equipe()