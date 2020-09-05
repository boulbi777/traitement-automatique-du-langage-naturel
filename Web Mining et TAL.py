#!/usr/bin/env python
# coding: utf-8

# Ce mini projet étant entièrement fait à l'aide de Jupyter-Notebook (bien plus adapté pour faire des comptes rendus), je vous prie d'évaluer ce rendu en consultant soit le fichier .html ou le notebook ipynb.
# En effet, ce fifier .py a été généré à partir du notebook.
  
# # Mini Projet : Web Mining et Traitement Automatique de Langues
# 
# ## Auteur : Boubacar TRAORE & Zakaria Jarraya


# In[1]:


# Changement de la police utilisée et de sa taille
from IPython.core.display import HTML, display
from tp_tools import change_font, plot_history
display(HTML(change_font()))


# In[2]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ## Importation des librairies nécessaires

# In[3]:


import json
import time
import spacy
import numpy as np
import pandas as pd
import newspaper as npp
import feedparser as fp
import itertools as it

from pprint import pprint
from collections import defaultdict

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go

init_notebook_mode(True)


# ## 1. Acquisition de données par écoute d’un flux RSS

# ### 1.1. Récupération des fichiers RSS

# #### Q1.

# In[4]:


#Lecture de la base de données déjà existente
with open('data/france-info-20200301.json', 'r') as f:
    articles = json.load(f)


# In[5]:


#type et taille de la base lue
type(articles), len(articles)


# In[6]:


print(len(list(articles.items())))


# In[7]:


# Affichage du premier élément du dictionnaire (le 1er article)
dict(list(articles.items())[0:1])


# Il s'agit d'un dictionnaire (venant d'un fichier json) contenant 660 clés associées chacune à des valeurs. Chaque clé correspond à l'URL d'un article extrait via le flux RSS. A chaque clé est associée une valeur qui est également un dictionnaire contenant comme information le titre, la date, l'auteur, la catégorie, le contenu et le lien image de l'article en question.

# Passons à la collecte de nouveaux articles via les flux RSS.

# In[8]:


# Stockage des urls 
urls = ['https://www.francetvinfo.fr/france.rss', 
        'https://www.francetvinfo.fr/europe.rss', 
        'https://www.francetvinfo.fr/entreprises.rss']


# In[9]:


print(urls[0].split('.rss')[0].split("/")[-1])


# Nous allons sélectionner le premier flux et jeter un coup d'oeil à son contenu.

# In[10]:


# on essaie le 1er feed
feed0 = fp.parse(urls[0])


# In[11]:


#overview of feed0
print(feed0.keys())


# Nous remarquons que ce flux retourne énormément d'informations. La clé 'entries' du dictionnaire correspond à la liste des articles trouvées. Nous pouvons également utiliser les informations annexes pour construire notre base de données.

# Nous allons maintenant extraire tous les flux et identifier les nouveaux articles qui ne sont pas déjà présent dans la base json dont on dispose déjà.

# In[12]:


new_articles_id = []
for url in urls:
    feed = fp.parse(url)
    for article in feed.entries:
        if article.id not in list(articles.keys()):
            new_articles_id.append([article.id, article.title]) #Ajout du lien du nouvel article


# In[13]:


new_articles = pd.DataFrame(new_articles_id, columns=["id", "title"])
print(new_articles.sample(3, random_state=42))


# In[14]:


# Voir les liens complets
new_articles.id.sample(3, random_state=42).tolist()


# In[15]:


len(new_articles_id)


# Nous pouvons constater qu'il y a 57 nouveaux articles tirés de ces 3 flux.

# ### 1.2. Récupéreation des articles

# #### Q2.

# Regardons le code source du premier article affiché après l'extraction des articles des flux RSS : https://www.francetvinfo.fr/sante/maladie/coronavirus/coronavirus-des-aperos-au-balcon-pour-vaincre-l-ennui-du-confinement_3878493.html#xtor=RSS-3-[france]

# Le code source de la page écite en HTML est très difficile à comprendre, les informations sont dispersées partout et on a du mal à s'y retrouver.

# Le package newspaper3k permet de faire de l'extraction automatique des contenus des articles. Il permet facilement de dissocier le titre, le contenu et plein d'autres information sur les sites d'articles. Sa prise en main est très facile. Prenons l'exemple d'un nouvel arcticle qu'on vient d'extraire.

# In[16]:


# Exemple du premier article extrait
article = npp.Article(new_articles.id[0])
article.download()
article.parse()
print(article.text)


# Comme nous pouvons le voir, le texte a été très bien extrait.

# #### Q3. et Q4.

# Nous allons d'abord définir des fonctions utiles à notre mise à jour.

# In[17]:


def extract_content(article_url):
    """Permet d'extraire l'auteur et contenu d'un article à partir de son url
    
    Arguments:
        article_url {str} -- Lien URL d'un article
    
    Returns:
        tuple of str -- Le tuple contient deux éléments. La 1ère est l'auteur et la 2e le contenu. 
    """
    article = npp.Article(article_url)
    article.download()
    article.parse()
    
    return article.authors, article.text


# In[18]:


def get_category(url):
    """ Retourne la catégorie d'un flux RSS de franceinfo à partir de son url
    
    Arguments:
        url {str} -- URL d'un flux RSS de franceinfo de type 'https://.../category.rss'
    
    Returns:
        str -- La catégorie du flux.
    """
    try :
        return urls[0].split('.rss')[0].split("/")[-1]
    except:
        return "" # Il ne s'agit pas d'URL de franceinfo


# In[19]:


def get_image_link(article):
    """ Permet d'obtenir le lien vers l'image d'un article de franceinfo
    
    Arguments:
        article {dict} -- Un article de franceinfo correspondant à un dictionnaire python.
    
    Returns:
        str -- Le lien URL vers l'image de l'article
    """
    try:
        if article.links[1].type == "image/jpeg": #le lien vers l'image existe
            return article.links[1].href
    except:
        pass
    
    return "" #there is no image file link


# Maintenant, passons à la définition de notre grande fonction de mise à jour de notre base de données. Elle fonctionne comme ci : 
# * Charge la base de données sous format json grâce au chemin d'accès donné en entrée
# * Ajoute à cette base de nouveaux articles trouvés via les flux RSS
# * Resauvegarde la base en format json dans le même répertoire donné

# In[20]:


def maj_database(database_path, urls):
    """Permet de mettre à jour la base de données des articles.
    
    Arguments:
        database_path {str} -- Chemin d'accès à la base de données au format json
        urls {list} -- List d'urls vers les flux RSS
    """
    
    # Lecture du fichier json
    try:
        with open('data/france-info-20200301.json', 'r') as f:
            articles = json.load(f)
    except: #stop
        return "Erreur, veuillez bien spécifier le chemin d'accès du fichier"
    
    
    #Initialisation des conteneurs d'informations
    date_times, authors, categories, contents, image_links = [], [], [], [], []
    
    # Collecte des articles venant des flux
    for url in urls:
        feed = fp.parse(url)
        for article in feed.entries:
            if article.id not in list(articles.keys()): #Mise à jour de la base
                
                # get author and text of the article
                author, content = extract_content(article.id)
                
                new_article = dict()
                new_article['title']      = article.title
                new_article['date']       = time.strftime('%Y-%m-%dT%H:%M:%S', article.published_parsed)
                new_article['author']     = author
                new_article['category']   = get_category(url)
                new_article['content']    = content
                new_article['image_link'] = get_image_link(article)
                
                #Ajouter du nouvel élément au json grâce à son "id"
                articles[article.id] = new_article
    
    #Sauvegarde du fichier avec la base mise à jour
    with open(database_path, 'w') as file:
        json.dump(articles, file)
    
    print('Done')


# Testons la fonction de mise à jour.

# In[21]:


# On revérifie bien que "articles" contient 660 éléments d'abord
print(len(articles))


# In[22]:


#Ensuite on fait la mise à jour...
maj_database(database_path='data/france-info-20200301.json', urls=urls)


# In[23]:


#On recharge articles et on regarde sa taille
with open('data/france-info-20200301.json', 'r') as f:
    articles = json.load(f)
print(len(articles))


# In[24]:


#Voyons le dernier élément pour s'assurer qu'il a bien été ajouté.
print(dict(list(articles.items())[-1:]))


# La base est effectivement mise à jour...
# Si on tente de recommencer la même maneouvre tout de suite à l'instant, la base n'augmentera probablement pas, puisque de nouveaux flux ne sont pas encore disponible.

# In[25]:


#Ensuite on fait la mise à jour...
maj_database(database_path='data/france-info-20200301.json', urls=urls)

#On recharge articles et on regarde sa taille
with open('data/france-info-20200301.json', 'r') as f:
    articles = json.load(f)
print(len(articles))


# Nous voyons bien que le contrôle d'identifiant est bien correct dans la base (pas d'ajout d'article qui existe déjà).

# ## 2. Extraction d'information

# ### 2.1. Extraction des entités nomées

# In[26]:


nlp = spacy.load("fr_core_news_sm")


# #### Q5.

# Spacy utilise des modèles de données pré-entrainés sur de gros corpus de documents dans différentes langues. Tout démarre de la labelisation de certaines entités nomées à la main suivi de certaines règles grammaticales définies au préalable selon la langue sélectionnée. Une fois ces exemples données au modèle, ce dernier est entrainé sur des réseaux de neurones pour apprendre à identifier les entités nommées bien taggées dans la base d'apprentissage. La détection des entités nommées dépend donc fortement de la base d'apprentissage, c'est pourquoi SpaCy donne la possibilité d'entrainer son propre modèle à partir d'observations qu'on peut labeliser nous même à la main en suivant une structure bien déterminé.

# A quoi correspondent les étiquettes IOB utilisées ? 

# In[27]:


document = nlp("Jean Dupont est maire de Plouguemeur. Apple n'y a pas de locaux.")
for token in document:
    print(token, token.ent_iob)


# Comme bien détaillé dans la documentation (https://spacy.io/api/annotation#iob), les tags IOB sont des entiers qui nous permettent de connaitre la place d'un token vis à vis d'une entité. Un token peut être à l'intérieur d'une entité nomée (I pour intérieur --> 1), à l'extérieur (O pour Outside -->) ou au début (B pour Begin --> 3). Dans notre exemple, l'entité nomée est bien "Jean" d'où l'entier 3 associé au texte "Jean". L'entité continue avec "Dupont" et tous les autres tokens ne font pas parti de l'entité.

# #### Q6.

# In[28]:


def get_named_entities(doc):
    """[summary]
    
    Arguments:
        doc {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    return [(entity.text, entity.label_) for entity in doc.ents]


# In[29]:


#Exemple d'article dont on veut connaitre les entités nomées
print(article.text)


# In[30]:


print(get_named_entities(nlp(article.text)))


# Six entités nommées ont été détectées dans ce texte. Le type de chaque entité est donné à sa droite. On sait donc que "Olivier" est une personne (PER) et "Chatenoy-en-Bresse" est un endroit (LOC comme Location). Quant au dernier type, la documentation officielle nous dit (MISC): "*Miscellaneous entities, e.g. events, nationalities, products or works of art.*"

# La documentation de Spacy (https://spacy.io/models/fr) indique qu'il y 4 entités nommées détectable par ce modèle : PER, LOC, ORG et MISC. Dans l'exemple que nous venons d'exécuter, il y en a 3. Nous allons garder les 4 types d'entités. ORG signifie une organisation (privée ou gouvernementale).

# #### Q7.

# Nous allons lancer ce calcul sur notre propre machine.

# In[31]:


result = []
for key in list(articles.keys())[:1]:
    doc = nlp(articles[key]['content'])
    for ent in doc.ents:
        result.append([ent.text, ent.start_char, ent.end_char, ent.label_])


# Le premier article prend en moyenne 22 millisecondes. Puisque nous avons 717 articles, la totalité du temps de traitement devrait être aux alentours de 16 secondes. Testons le.

# In[32]:


result = []
for key in articles.keys():
    doc = nlp(articles[key]['content'])
    for ent in doc.ents:
        result.append([ent.text, ent.start_char, ent.end_char, ent.label_])


# Le temps moyen est de 28 secondes.

# ### 2.2. Analyse de entités nommées

# #### Q8.

# Nous allons directement travailler avec les 717 articles (base déjà mise à jour).

# In[33]:


def plot_bar(series, title, top_N = 10):
    """ Fonction permettant d'afficher un bar plot horizontal à partir d'une serie pandas.
    
    """
    
    series = series[::-1][-top_N-1:-1]
    to_plot  = [go.Bar(
                x=series.values,
                y=series.index,
                orientation = 'h'
    )]
    
    layout = go.Layout(title=title)
    fig = go.Figure(data=to_plot, layout=layout)
    iplot(fig)


# In[34]:


def get_top_entities(collection, top_N = 20):
    """[summary]
    
    Arguments:
        collection {list} -- La liste des documents text du corpus
    
    Keyword Arguments:
        top_N {int} -- Le nombre d'entités à afficher et à sauvegarder (default: {20})
    
    Returns:
        list of pandas Series -- La liste des top_N entités ordonnés par occurence pour chaque type d'entité
    """
    
    entities_infos = []
    
    for article_content in collection:
        doc = nlp(article_content)
        for entity in doc.ents:
            entities_infos.append([entity.text, entity.label_])
    df = pd.DataFrame(entities_infos, columns=["entity_name", "entity_type"])
    saved_entities = []
    for _type in df.entity_type.unique():
        _series = df[df.entity_type == _type].entity_name.value_counts().sort_values(ascending=False)
        plot_bar(_series, title = _type, top_N=top_N)
        saved_entities.append(_series[:top_N+1])
    return saved_entities


# In[36]:


# Création d'une collection totale des contenus articles
collection = []
for key in articles.keys():
    collection.append(articles[key]['content'])


# In[37]:


saved_entities = get_top_entities(collection)


# #### Q9.

# Nous écrivons une fonction qui retourne le nombre de co-occurrence de deux entités dans un document donné. On considère qu'il y a co-occurrence entre deux entité dans un document lorsque les deux entités se succèdent dans la liste exhaustive des entités. Il faut comprendre par cela que les deux entités ne sont pas forcement collés l'un à l'autre dans le texte brut du document. Prenons un exemple dans notre collection :

# In[38]:


print(collection[47][:300] + "...")


# En observant le 47e document de notre collection, on observe que l'entité "General Electric" et "Belfort" se succèdent dans la liste des entités de ce document. Pourtant la préposition "à" sépare ces deux entitiés à chaque fois mais on ne tient pas compte de cette préposition dans le calcul du nombre de co-occurence des deux entités. On voit donc bien qu'il y a 2 co-occurences de ces entités dans ce début de texte...

# In[39]:


def get_entities_cooccurrence(e1, e2, doc):
    """[summary]
    
    Arguments:
        e1 {spaCy entity} -- La 1ère entité nomée
        e2 {spaCy entity} -- La 2eme entité nomée
        doc {spaCy doc}   -- Un document spaCy où chercher les co-occurrences
    
    Returns:
        int -- Le nombre total de co-occurence dans le document donné
    """
    nb = 0 #nombre total de cooccurrences
    entities = doc.ents
    for i in range(len(entities)-1):
        if (entities[i].text == e1 and entities[i+1].text == e2) or            (entities[i].text == e2 and entities[i+1].text == e1):
            nb += 1
    
    return nb


# Jetons un oeil sur les entités nommées du document 47 de notre collection.

# In[40]:


print(get_named_entities(nlp(collection[47])))


# In[41]:


# Cherchons le nombre de co-occurrence des deux entités nomées précédemment citées
print(get_entities_cooccurrence('General Electric', 'Belfort', nlp(collection[47])))


# #### Q10.

# Cherchons des co-occurences parmi la liste des entités nommées les plus fréquentes

# In[42]:


# Voyons voir le contenu du premier type d'entité sauvegardé (PER)
print(saved_entities[0])


# Il s'agit d'une pandas Series dont l'index est constitué des entités nomées et les valeurs sont le nombre d'occurence. On s'interessera donc aux index pour le calcul des co-occurences.

# In[43]:


#Stockage des articles de notre collection comme document nlp
docs = []
for doc in collection:
    docs.append(nlp(doc))


# In[44]:


occ_infos = defaultdict(int) #dictionnaire qui contiendra toutes les coocurrences des entités
for doc in docs:
    for entity_type_1, entity_type_2 in it.combinations(saved_entities, 2):
        for entity_1 in entity_type_1.index.values:
            for entity_2 in entity_type_2.index.values:
                occ_infos[entity_1 + " -- " + entity_2] += get_entities_cooccurrence(entity_1, entity_2, doc)


# In[45]:


print(pd.Series(occ_infos).sort_values(ascending=False)[:40])


# In[46]:


plot_bar(pd.Series(occ_infos).sort_values(ascending=False), 
         title="Cooccurrence des entités nommées les plus fréquentes",
         top_N=23)


# In[47]:


print(nlp("Brexit").ents[0].label_, nlp("UE").ents[0].label_, nlp("Union Européenne").ents[0].label_)


# "Brexit" est identifié comme une personne tandis que "UE" et "Union Européenne". Nous pouvons remarquer que le sujet émergent est le "Brexit" et cette entité est intimement liée à une organisation (UE) ou à un endroit (UK ou Londres). Les relations semblent assez pertinentes.

# #### Q11.

# In[48]:


def characterize_link(ent_list, docs):
    link = defaultdict(list)
    for i in range(len(docs)):
        doc_index = defaultdict(list)
        for e1, e2 in ent_list:
            for j in range(len(docs[i].ents) - 1):
                #Si les deux entités sesuivent
                if e1 == docs[i].ents[j].text and e2 == docs[i].ents[j+1].text:
                    # Capter le texte entre les deux entités
                    between_doc = docs[i].text[docs[i].ents[j].end_char : docs[i].ents[j+1].start_char]
                    for token in nlp(between_doc): #vérifier s'il y a un verbe entre les deux 
                        if token.pos_ == 'VERB':
                            doc_index[i].append(token)
                    if len(doc_index[i])>0:        
                        link[e1 + " -- " + e2].append(dict(doc_index))
    return link  


# In[49]:


find_links = [
    ("Brexit", "UE"), 
    ("France", "JT"), 
    ("Chine", "Covid-19")
]


# In[50]:


print(pprint(dict(characterize_link(find_links, docs=docs))))


# Comme nous pouvons constater le résultat, cette fonction nous permet de donner en entrée une liste de tuples d'entitées et retourne pour chaque tuple le numéro du document dans lequel il a été trouvé suivi de la liste de tous les verbes qui caractérisent la liaison entre ces deux entités. Par exemple, en regardant le tuple (Chine, Covid19), nous pouvons constater que les verbes les liant dans le document 602 sont "avoir", "hospitaliser" et "tester". Jetons un coup d'oeil à ce document.

# In[51]:


print(docs[602])


# Ces verbes trouvés sont essentiellement dans le 1er paragraphe.

# #### Q12.

# In[52]:


def get_entity_pairs(verb, docs):
    pairs = []
    for doc in docs:
        for i in range(len(doc.ents) - 1):
            e1, e2 = doc.ents[i], doc.ents[i+1]
            between_doc = doc.text[e1.end_char : e2.start_char]
            for token in nlp(between_doc): #vérifier s'il y a un verbe entre les deux
                #print(token.text)
                if token.pos_ == 'VERB' and token.text == verb:
                    pairs.append((e1, e2))
    return pairs


# In[53]:


print(get_entity_pairs("testé", docs[300:700]))


# Cet exemple montre que la nature de la relation définie uniquement par un verbe entre les deux entités n'est pas très bonne. On a pu retrouver l'exmple donné par le tuple (Chine, Covid-19) dans la question 11 mais les autres réponses ne sont pas assez pertinentes. Il faudrait plutot une analyse syntaxique plus approfondie pour s'attendre à des résultats plus convaincants, ceci permettrait de mieux définir la nature des relations entre les entités.

# In[54]:


print(get_entity_pairs("juger", docs))


# Cet exemple reste quand même pas mal, les entités retournées ont un lien avec le verbe recherché.
