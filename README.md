# Aurélien - Data Scientist & AI Engineer

Bonjour ! Je suis **Aurélien**, 29 ans, Data Scientist et Ingénieur IA passionné par l'intelligence artificielle et les nouvelles technologies. Je développe des solutions innovantes en Machine Learning, Deep Learning et Computer Vision, avec une expertise particulière en **GANs**, **Computer Vision** et **analyse de données scientifiques**.

[![Website](https://img.shields.io/badge/Website-aurel--test.fr-blue?style=flat-square)](https://aurel-test.fr)
[![GitHub followers](https://img.shields.io/github/followers/aureltest?style=flat-square&logo=github)](https://github.com/aureltest)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

---

## 🎯 À Propos

Diplômé **Ingénieur IA** (Bac+5, OpenClassrooms), je combine des compétences techniques solides en Data Science avec une passion pour résoudre des problèmes complexes grâce à l'IA. Mon portfolio inclut des projets variés allant de la **génération d'images** au **tracking de mains en temps réel**, en passant par l'**analyse spectroscopique** et la **classification d'œuvres d'art**.

---

## 💼 Expertise & Compétences

### 🤖 Intelligence Artificielle & Deep Learning

**Architectures de Réseaux de Neurones**
- **Generative Adversarial Networks (GANs)**: DCGAN, Conditional GANs
- **Convolutional Neural Networks (CNNs)**: Classification, détection, segmentation
- **Transfer Learning**: Fine-tuning de modèles pré-entraînés (CLIP, ResNet, VGG, BERT)
- **Transformers**: BERT, GPT, Vision Transformers (ViT)

**Computer Vision Avancée**
- Classification d'images multi-classes
- Détection et tracking d'objets en temps réel
- Génération d'images synthétiques (GANs)
- Segmentation sémantique
- Analyse et clustering d'œuvres d'art
- Hand tracking et pose estimation

**Natural Language Processing**
- Traitement et classification de textes
- Analyse de sentiment et détection de bad buzz
- Embeddings textuels (Word2Vec, BERT, sentence-transformers)
- Named Entity Recognition (NER)

**Apprentissage Contrastif & Embeddings**
- Supervised Contrastive Learning
- Réduction dimensionnelle (t-SNE, UMAP, PCA)
- Recherche de similarité avec FAISS
- Clustering dans l'espace latent

### 📊 Data Science & Machine Learning Classique

**Algorithmes & Techniques**
- Classification supervisée (Random Forest, SVM, XGBoost, LightGBM)
- Régression linéaire et non-linéaire
- Clustering (K-means, DBSCAN, Hierarchical, GMM)
- Dimensionality Reduction (PCA, LDA, t-SNE, UMAP)
- Ensemble Methods (Bagging, Boosting, Stacking)

**Applications Métier**
- Segmentation client et RFM analysis
- Modèles de scoring et credit scoring
- Prédiction de churn
- Systèmes de recommandation (collaborative filtering, content-based)
- Séries temporelles et forecasting
- A/B Testing et analyse d'expériences

**Analyse de Données**
- Exploratory Data Analysis (EDA)
- Feature engineering et sélection de variables
- Traitement des valeurs manquantes et outliers
- Normalisation et standardisation
- Analyse statistique et tests d'hypothèses

### 🔬 Domaines Scientifiques Spécialisés

**Spectroscopie & Traitement de Signaux**
- Classification de spectres LIBS (Laser-Induced Breakdown Spectroscopy)
- Analyse de données haute dimensionnalité
- Prétraitement de signaux physiques
- Machine Learning pour données scientifiques
- Applications en géologie, métallurgie, forensique

**Computer Vision pour l'Art**
- Classification d'œuvres d'art par style/artiste
- Clustering non supervisé d'images artistiques
- Extraction de features visuelles
- Analyse de collections muséales

### 💻 Développement & Engineering

**Backend & Web Development**
- **Python**: Flask, Django
- **PHP**: Symfony
- **APIs**: RESTful, FastAPI
- **Bases de données**: SQL, MySQL, PostgreSQL, SQLite, Neo4j

**DevOps & MLOps**
- Git/GitHub (version control, collaboration)
- Docker (containerisation)
- CI/CD pipelines
- Environnements virtuels (venv, conda)
- Model versioning et monitoring
- Déploiement de modèles ML

**Langages de Programmation**
```
Python (Expert) • SQL (Avancé) • PHP • Java • HTML/CSS
```

---

## 🚀 Projets Phares

### 🎨 [Classification d'Œuvres d'Art avec CLIP](https://github.com/aureltest/clustering-artwork)
**Oct 2025 | Computer Vision • Transfer Learning • Contrastive Learning**

Fine-tuning du modèle CLIP pour classifier **6,280 œuvres d'art** en 5 catégories stylistiques avec une précision de **99.5%**. Le projet combine apprentissage contrastif supervisé et clustering non supervisé pour créer un espace latent sémantiquement cohérent.

**Approches implémentées:**
- Supervised Contrastive Learning pour optimiser les embeddings
- Transfer Learning sur CLIP pré-entraîné
- Clustering K-means dans l'espace latent
- Visualisation UMAP/t-SNE des représentations

**Stack**: PyTorch, OpenAI CLIP, Supervised Contrastive Learning, UMAP, t-SNE, TensorBoard, scikit-learn

**Résultats**: 99.5% accuracy, séparation claire des clusters, embeddings sémantiques de haute qualité

---

### 🧑 [Générateur de Visages avec DCGAN](https://github.com/aureltest/face_generator)
**Août 2025 | Deep Learning • GANs • Computer Vision**

Implémentation d'un **Deep Convolutional GAN (DCGAN)** pour générer des visages humains photoréalistes qui n'existent pas dans la réalité. Le modèle apprend à créer de nouvelles images de visages en capturant les distributions statistiques du dataset d'entraînement.

**Architecture:**
- Generator: Convolutions transposées, BatchNorm, activation ReLU/Tanh
- Discriminator: CNN avec Leaky ReLU, discrimination binaire réel/faux
- Training adversarial avec loss minimax

**Dataset**: CelebA (célébrités, 200K+ images)

**Stack**: Python, PyTorch/TensorFlow, Deep Learning, Computer Vision

**Applications**: Augmentation de données, génération de datasets synthétiques, recherche en deep learning génératif

---

### 👋 [Module de Hand Tracking Temps Réel](https://github.com/aureltest/HandTrackingModule)
**Déc 2024 | Computer Vision • Real-time Processing • OpenCV**

Module Python réutilisable pour le **tracking de mains en temps réel** utilisant MediaPipe et OpenCV. Détection de 21 landmarks par main avec reconnaissance de gestes (doigts levés/baissés).

**Fonctionnalités:**
- Détection multi-mains via webcam
- Tracking de 21 points de repère (landmarks)
- Reconnaissance de gestes (pinch, grab, swipe)
- Module plug-and-play pour projets CV

**Stack**: Python, OpenCV, MediaPipe, Real-time Processing

**Use Cases**: Contrôle de volume gestuel, interface sans contact, reconnaissance de signes, contrôle de souris virtuel

---

### 🔬 [Classification LIBS Spectroscopique](https://github.com/aureltest/LIBS-classification)
**Nov 2024 | Machine Learning • Signal Processing • Scientific Computing**

Système de **classification automatique de spectres LIBS** (Laser-Induced Breakdown Spectroscopy) pour l'identification de matériaux. Application ML à des données scientifiques haute dimensionnalité pour l'analyse élémentaire.

**Domaine d'application:**
- Géologie et exploration minérale
- Industrie sidérurgique et métallurgique
- Science forensique
- Contrôle qualité industriel
- Analyse environnementale

**Techniques:**
- Prétraitement de spectres (normalisation, baseline correction)
- Feature extraction et réduction de dimensionnalité (PCA)
- Classification supervisée (SVM, Random Forest, XGBoost)
- Deep Learning pour spectres 1D (CNN 1D)
- Analyse multivariée (PLS-DA, LDA)

**Stack**: Python, scikit-learn, scipy, NumPy, Machine Learning, Signal Processing

**Challenges résolus**: Gestion de la haute dimensionnalité, variabilité spectrale, interférences entre pics élémentaires

---

### 🛡️ [Détection de Bad Buzz sur Réseaux Sociaux](https://github.com/aureltest)
**NLP • Deep Learning • Sentiment Analysis**

Système de **détection automatique de bad buzz** utilisant NLP et Deep Learning pour identifier les crises réputationnelles émergentes sur les réseaux sociaux en temps réel.

**Pipeline:**
- Collecte de données Twitter/Reddit
- Prétraitement NLP (tokenization, lemmatization, stopwords)
- Embeddings textuels (BERT, sentence-transformers)
- Classification de sentiment multi-classes
- Détection d'anomalies et alertes

**Stack**: Python, BERT, Transformers, NLP, Deep Learning, Sentiment Analysis

**Métriques**: Precision, Recall, F1-Score, Confusion Matrix

---

### 🛒 [Segmentation Client E-commerce](https://github.com/aureltest)
**Customer Analytics • Clustering • Business Intelligence**

Analyse et **segmentation de clients e-commerce** pour personnalisation marketing et optimisation des ventes. Identification de segments clients à forte valeur ajoutée.

**Méthodologie:**
- RFM Analysis (Recency, Frequency, Monetary)
- Feature engineering (lifetime value, panier moyen, taux de retour)
- Clustering K-means, DBSCAN, Hierarchical
- Profiling et caractérisation des segments
- Recommandations business actionnables

**Stack**: Python, Pandas, scikit-learn, K-means, Matplotlib, Seaborn

**Résultats**: Identification de 5-7 segments distincts, stratégies marketing ciblées, augmentation du ROI

---

### 💳 [Modèle de Scoring Machine Learning](https://github.com/aureltest)
**Credit Scoring • Risk Assessment • Classification**

Développement de **modèles de scoring prédictifs** pour évaluation de risque crédit et aide à la décision financière.

**Approche:**
- Préparation et nettoyage de données financières
- Feature engineering (ratios financiers, historiques)
- Gestion du déséquilibre de classes (SMOTE, undersampling)
- Entraînement de modèles (Logistic Regression, XGBoost, Random Forest)
- Calibration de probabilités
- Interprétabilité (SHAP values, feature importance)

**Stack**: Python, scikit-learn, XGBoost, SHAP, imbalanced-learn

**Métriques**: AUC-ROC, Precision-Recall curve, KS statistic, Gini coefficient

---

### 🎬 [Système de Recommandation de Contenu](https://github.com/aureltest)
**Recommendation Systems • Collaborative Filtering • Matrix Factorization**

Application de **recommandation personnalisée** utilisant algorithmes de filtrage collaboratif et content-based filtering.

**Techniques:**
- Collaborative Filtering (User-based, Item-based)
- Matrix Factorization (SVD, ALS)
- Content-based filtering avec embeddings
- Hybrid approaches combinant plusieurs méthodes
- Cold start problem mitigation

**Stack**: Python, scikit-surprise, TensorFlow, Collaborative Filtering

**Métriques**: RMSE, MAE, Precision@K, Recall@K, NDCG

---

## 🎓 Parcours Formation - Ingénieur IA

### Projets de la Formation OpenClassrooms (Bac+5)

Série de **10 projets professionnalisants** couvrant l'ensemble du spectre de l'IA moderne, du Machine Learning classique au Deep Learning avancé, avec déploiement sur cloud Azure.

#### 📚 [Projet 10 - Capstone Project IA](https://github.com/aureltest)
**Juil 2024 | Intégration complète • MLOps • Production**
- Projet final intégrant toutes les compétences acquises
- Développement end-to-end d'une solution IA
- Déploiement en production avec monitoring
- MLOps et maintenance de modèles

#### 🧠 [Projet 9 - Deep Learning Avancé](https://github.com/aureltest)
**Technologies**: PyTorch/TensorFlow, Transfer Learning, Fine-tuning

#### 🔍 [Projet 8 - Computer Vision Avancée](https://github.com/aureltest)
**Nov 2023**
- Architectures CNN complexes
- Détection et segmentation d'objets
- Techniques d'optimisation avancées

#### 📝 [Projet 7 - Natural Language Processing](https://github.com/aureltest)
**Nov 2023**
- Classification de textes multi-classes
- Embeddings et représentations textuelles
- Transformers et attention mechanisms

#### 📊 [Projet 6 - Séries Temporelles](https://github.com/aureltest)
**Juil 2023**
- Forecasting et prédiction
- ARIMA, LSTM, Prophet
- Analyse de tendances et saisonnalité

#### 🎯 [Projet 5 - Clustering & Segmentation](https://github.com/aureltest)
**Juil 2023**
- Apprentissage non supervisé
- K-means, DBSCAN, Hierarchical clustering
- Réduction de dimensionnalité

#### 📈 [Projet 4 - Régression Avancée](https://github.com/aureltest)
**Juil 2023**
- Feature engineering poussé
- Régularisation (Ridge, Lasso, ElasticNet)
- Ensemble methods

#### 🏆 [Projet 3 - Classification Multi-classes](https://github.com/aureltest)
**Juil 2023**
- Random Forest, Gradient Boosting
- Feature selection et importance
- Cross-validation stratifiée

#### 🔬 [Projet 2 - Machine Learning Fondamental](https://github.com/aureltest)
**Nov 2022**
- EDA et préparation de données
- Premiers modèles supervisés
- Évaluation et métriques

---

## 🛠️ Stack Technique Complète

### Intelligence Artificielle & Deep Learning
```python
PyTorch • TensorFlow • Keras • Hugging Face Transformers
OpenAI CLIP • BERT • GPT • Vision Transformers
scikit-learn • XGBoost • LightGBM • CatBoost
```

### Computer Vision
```python
OpenCV • MediaPipe • PIL/Pillow • torchvision
YOLO • Detectron2 • Mask R-CNN
Image Augmentation • Object Detection • Segmentation
```

### Natural Language Processing
```python
NLTK • spaCy • Transformers • sentence-transformers
BERT • GPT • Word2Vec • FastText • GloVe
TextBlob • Gensim • Tokenizers
```

### Data Science & Analyse
```python
NumPy • Pandas • Matplotlib • Seaborn • Plotly
SciPy • Statsmodels • scikit-learn
UMAP • t-SNE • PCA • FAISS
```

### Outils de Développement
```python
Jupyter Notebooks • Google Colab • VS Code
TensorBoard • Weights & Biases • MLflow
Git/GitHub • Docker • Linux/Bash
```

### Bases de Données
```sql
MySQL • PostgreSQL • SQLite • SQL Server
Neo4j • MongoDB • Redis
Pandas (data manipulation) • SQLAlchemy
```

### Web Development
```python
Flask • Django • FastAPI • Streamlit
Symfony (PHP) • REST APIs
HTML/CSS • JavaScript • Bootstrap
```

### Cloud & Déploiement
```yaml
Microsoft Azure ML • AWS (basics)
Docker • Kubernetes (basics)
CI/CD • Model Serving • API Deployment
```

---

## 📈 Méthodologie de Travail

### 🔄 Cycle de Vie d'un Projet ML/IA

```
1. 📋 Compréhension du Problème
   └─ Définition des objectifs, KPIs, contraintes métier

2. 🔍 Exploration des Données (EDA)
   └─ Visualisation, statistiques, identification de patterns

3. 🧹 Preprocessing & Feature Engineering
   └─ Nettoyage, transformation, création de features

4. 🤖 Modélisation
   └─ Sélection d'algorithmes, entraînement, validation croisée

5. 📊 Évaluation & Optimisation
   └─ Métriques, hyperparameter tuning, ensemble methods

6. 🚀 Déploiement & Monitoring
   └─ API, containerisation, monitoring de performance

7. 📝 Documentation & Communication
   └─ README, notebooks, présentation des résultats
```

### ✅ Bonnes Pratiques

**Code Quality**
- ✔️ Versioning avec Git et branches de développement
- ✔️ Code modulaire et réutilisable
- ✔️ Docstrings et commentaires explicatifs
- ✔️ Notebooks Jupyter organisés et annotés
- ✔️ Virtual environments pour isolation des dépendances

**Expérimentation ML**
- ✔️ Tracking des expériences (TensorBoard, W&B)
- ✔️ Seed fixing pour reproductibilité
- ✔️ Validation croisée systématique
- ✔️ Test set séparé et jamais touché avant la fin
- ✔️ Métriques multiples pour évaluation robuste

**Documentation**
- ✔️ README détaillés avec contexte et résultats
- ✔️ Requirements.txt et environment.yml
- ✔️ Instructions de reproduction
- ✔️ Visualisations des résultats
- ✔️ Notebooks didactiques

---

## 📚 Formation & Certifications

### 🎓 Diplômes
**Ingénieur Intelligence Artificielle** - OpenClassrooms (2022-2024)
- Certification RNCP Niveau 7 (Bac+5) - "Data Scientist"
- 10 projets professionnalisants validés
- Compétences: ML, DL, Computer Vision, NLP, MLOps, Cloud Azure

### 📜 Certifications Techniques
- **Machine Learning** - Expertise en algorithmes supervisés/non supervisés
- **Deep Learning** - CNNs, GANs, Transformers, Transfer Learning
- **Computer Vision** - Classification, détection, segmentation, génération
- **NLP** - Traitement du langage, embeddings, transformers
- **Cloud Computing** - Microsoft Azure ML, déploiement de modèles

### 🔧 Développement Web & Software Engineering
- **Full-Stack Development** - Python (Flask, Django), PHP (Symfony)
- **Bases de Données** - SQL, MySQL, PostgreSQL, Neo4j, PowerBI
- **DevOps** - Git, Docker, CI/CD pipelines
- **APIs** - RESTful, FastAPI, microservices

### 📖 Auto-formation Continue
- Veille technologique active sur les dernières avancées IA
- Participation à des compétitions Kaggle
- Lecture de papers de recherche (arXiv, NeurIPS, CVPR, ICML)
- Expérimentation avec les modèles émergents (Diffusion Models, LLMs, Vision Transformers)

---

## 🎯 Domaines d'Application

### 🖼️ Computer Vision
- Classification d'images multi-classes
- Détection d'objets (YOLO, Faster R-CNN)
- Segmentation sémantique et d'instances
- Génération d'images (GANs, Diffusion Models)
- Hand/Pose tracking temps réel
- Analyse d'œuvres d'art et patrimoine culturel

### 📝 Natural Language Processing
- Classification et catégorisation de textes
- Analyse de sentiment et détection d'opinions
- Named Entity Recognition (NER)
- Chatbots et assistants conversationnels
- Résumé automatique de textes
- Détection de bad buzz sur réseaux sociaux

### 🛍️ Recommender Systems
- Filtrage collaboratif (User-based, Item-based)
- Content-based filtering
- Hybrid recommendation systems
- Cold start problem solutions
- Personnalisation e-commerce et contenu

### 👥 Customer Analytics & Business Intelligence
- Segmentation client (RFM, clustering)
- Prédiction de churn et rétention
- Customer Lifetime Value (CLV)
- Scoring et credit risk assessment
- A/B Testing et optimisation
- Tableaux de bord et dataviz

### 🔬 Applications Scientifiques
- Spectroscopie computationnelle (LIBS)
- Analyse de signaux physiques
- Classification de données haute dimensionnalité
- Traitement de données chimiques/physiques
- Machine Learning pour sciences expérimentales

### 🌐 Web Applications & APIs
- Développement full-stack Python/PHP
- APIs RESTful pour modèles ML
- Déploiement de modèles en production
- Applications web avec Flask/Django
- Microservices et architecture cloud

---

## 💡 Centres d'Intérêt Tech

### 🔥 Technologies Émergentes
- **Generative AI**: Stable Diffusion, Midjourney, DALL-E, LLMs
- **Large Language Models**: GPT-4, Claude, LLaMA, open-source LLMs
- **Vision Transformers**: ViT, CLIP, DINO, SAM
- **Diffusion Models**: Stable Diffusion, ControlNet, LoRA fine-tuning
- **Multimodal AI**: CLIP, BLIP, LLaVA, vision-language models
- **Robotics** : Reachy Mini (Pollen Robotics), ROS

### 📚 Veille Technologique
- Papers de recherche en IA (arXiv, conferences)
- Blogs techniques (Distill, Towards Data Science)
- Competitions Kaggle et challenges ML
- Open-source contributions et expérimentations

---

## 📫 Contact & Liens

- 🌐 **Site Web**: [aurel-test.fr](https://aurel-test.fr)
- 💻 **GitHub**: [github.com/aureltest](https://github.com/aureltest)
- 📍 **Localisation**: France
- 💼 **Statut**: Employé dans un centre d'innovation (https://le-click.be/)

---

## 🚀 Actuellement

🔭 **Je travaille sur**: De l'analyse d'émission spectroscopique (LIBS)
🌱 **J'apprends**: Actuellement j'apprends à utiliser le ROS
👯 **Je cherche à collaborer sur**: Projets open-source en IA/ML  
💬 **Demandez-moi à propos de**: PyTorch, Computer Vision, GANs, ML deployment  
⚡ **Fun fact**: Mon setup inclut 3 écrans, 1 lapin, et beaucoup trop de café ☕

---

<div align="center">

### ⭐ Si mes projets vous intéressent, n'hésitez pas à mettre une étoile !

**Toujours en quête de nouveaux défis en Data Science et Intelligence Artificielle !**

![Profile Views](https://komarev.com/ghpvc/?username=aureltest&color=brightgreen)

</div>
