# sc21_projects

# 3 - Validation, Feature Selection, and Hyperparameter Optimization
Реализация различных схем кросс-валидации с нуля (K-Fold, Stratified, Time Series Split) и отбора признаков (Permutation Importance); ознакомление с SHAP. Оптимизация гиперпараметров с помощью optuna.

# 4 - Binary Classification
Построение моделей для бинарной классификации (источник данных - kaggle.com). Собственные версии Logistic Regression, KNN и Naive Bayes с нуля, включая расчет градиентов и SGD; Использован CountEncoder, метрики auc-roc и auc-pr.

# 5 - Tree ensembles
Реализация Decision Tree, Random Forest и Gradient Boosting с нуля, включая обработку категориальных признаков с предотвращением утечки данных и временное разделение выборки. LightGBM, CatBoost и XGBoost, исследование особенностей каждого бустинга и оценка переобучения.

# 6 - Dimensionality Reduction
Разреженные матрицы, методы снижения размерности (PCA, UMAP, t-SNE, LLE), SVD для задач сжатия изображений (датасет MNIST) и выделения фона в видео, ранг аппроксимации и качество реконструкции.

# 7 - Clusterization
Методы кластеризации (KMeans, DBSCAN, Agglomerative Clustering, Gaussian Mixture Models) - реализация с нуля (кроме Agglomerative), сравнение с оригинальными из sklearn. 

# 8 - Multilayer Perceptrone from scratch
MLP, PyTorch, Adam optimizer
Имплементация модели MLP - через SGD и с использованием оптимизатора ADAM; основы работы с PyTorch; анализ эффективности различных функций активации.

# 9 - Convolutional Neural Networks
CNN, GPU, albumentations, TTA
Имплементация архитектуры CNN для задачи мультиклассовой классификации жестов по фотографиям с использованием слоев сверток и пулинга PyTorch; применение аугментаций (albumentations, TTA) для улучшения предсказания модели; чтение изображений реализовано через .npy формат для обучения на GPU (ресурсы Kaggle Notebook); работа с дообучением pretrained-моделей из библиотеки timm.

# 10 - RNN
Реализация архитектур RNN с нуля (Vanilla RNN, GRU, LSTM) для задачи генерации текста и классификации пола имен. Мультитаск-обучение под несколько задач; использование pytorch hook для наблюдения за градиентами.

# 11 - RNN with cross-attention and Transformers with self-attention 
Реализация механизма перекрестного внимания в архитектурах RNN для задачи машинного перевода (англ > рус).
Повтор архитектуры трансформеров с самовниманием из Attention is All You Need.
