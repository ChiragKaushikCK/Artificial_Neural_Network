# Artificial_Neural_Network

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</div>

<h1 align="center" style="color: #6200EE; font-family: 'Segoe UI', sans-serif; font-size: 3.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
  <br>
  <a href="https://github.com/your-username/artificial-neural-network-classification">
    <img src="https://placehold.co/600x200/6200EE/FFFFFF?text=Artificial+Neural+Network" alt="ANN Banner" style="border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
  </a>
  <br>
  🧠 Artificial Neural Network (ANN) for Classification 🚀
  <br>
</h1>

<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Dive into the world of **Artificial Neural Networks** with this comprehensive Jupyter Notebook! This project demonstrates the end-to-end process of building, training, and evaluating a Dense Neural Network for a classification task (likely customer churn prediction). Learn essential concepts like data preprocessing, feature scaling, one-hot encoding, model architecture design with Keras, training, and performance evaluation using a confusion matrix. Ideal for anyone looking to build a strong foundation in deep learning! 💡
</p>

<br>

<details style="background-color: #EDE7F6; border-left: 5px solid #6200EE; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 700px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <summary style="font-size: 1.3em; font-weight: bold; color: #333; cursor: pointer;">Table of Contents</summary>
  <ol style="list-style-type: decimal; padding-left: 25px; line-height: 1.8;">
    <li><a href="#about-the-project" style="color: #6200EE; text-decoration: none;">📚 About The Project</a></li>
    <li><a href="#dataset" style="color: #6200EE; text-decoration: none;">📦 Dataset</a></li>
    <li><a href="#modeling-approach" style="color: #6200EE; text-decoration: none;">⚙️ Modeling Approach</a></li>
    <li><a href="#features" style="color: #6200EE; text-decoration: none;">🎯 Features</a></li>
    <li><a href="#prerequisites" style="color: #6200EE; text-decoration: none;">🛠️ Prerequisites</a></li>
    <li><a href="#how-to-run" style="color: #6200EE; text-decoration: none;">📋 How to Run</a></li>
    <li><a href="#example-output" style="color: #6200EE; text-decoration: none;">📈 Example Output</a></li>
    <li><a href="#code-breakdown" style="color: #6200EE; text-decoration: none;">🧠 Code Breakdown</a></li>
    <li><a href="#contribute" style="color: #6200EE; text-decoration: none;">🤝 Contribute</a></li>
  </ol>
</details>

---

<h2 id="about-the-project" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  📚 About The Project
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This project focuses on building an **Artificial Neural Network (ANN)** to tackle a classification problem. It provides a hands-on experience in using popular deep learning libraries like Keras (built on TensorFlow) alongside scikit-learn for data preparation. The notebook guides you through the crucial steps involved in a typical machine learning pipeline for neural networks:
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #EDE7F6; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #6200EE;">Data Loading & Preprocessing:</strong> Handling missing values, encoding categorical features (One-Hot Encoding).
  </li>
  <li style="margin-bottom: 10px; background-color: #EDE7F6; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #6200EE;">Feature Scaling:</strong> Applying Standardization to numerical features.
  </li>
  <li style="margin-bottom: 10px; background-color: #EDE7F6; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #6200EE;">Train-Test Split:</strong> Preparing data for model training and evaluation.
  </li>
  <li style="margin-bottom: 10px; background-color: #EDE7F6; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #6200EE;">ANN Architecture:</strong> Designing a Sequential model with Dense layers.
  </li>
  <li style="margin-bottom: 10px; background-color: #EDE7F6; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #6200EE;">Model Training & Evaluation:</strong> Compiling, fitting, predicting, and assessing performance using accuracy and confusion matrix.
  </li>
</ul>

---

<h2 id="dataset" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  📦 Dataset
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  The project likely uses a dataset commonly found in machine learning tutorials, such as a **Customer Churn dataset**, to predict if a customer will exit a service. The dataset includes various features that might influence churn. An example of the initial data structure is shown below:
</p>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>      <span style="color: #9CDCFE;">RowNumber</span>  <span style="color: #9CDCFE;">CustomerId</span>    <span style="color: #9CDCFE;">Surname</span>  <span style="color: #9CDCFE;">CreditScore</span>  <span style="color: #9CDCFE;">Age</span>  <span style="color: #9CDCFE;">Tenure</span>    <span style="color: #9CDCFE;">Balance</span>  <span style="color: #9CDCFE;">NumOfProducts</span>  <span style="color: #9CDCFE;">HasCrCard</span>  <span style="color: #9CDCFE;">IsActiveMember</span>  <span style="color: #9CDCFE;">EstimatedSalary</span>  <span style="color: #9CDCFE;">Exited</span>
<span style="color: #B5CEA8;">0</span>             <span style="color: #B5CEA8;">1</span>    <span style="color: #B5CEA8;">15634602</span>   <span style="color: #CE9178;">Hargrave</span>          <span style="color: #B5CEA8;">619</span>   <span style="color: #B5CEA8;">42</span>       <span style="color: #B5CEA8;">2</span>       <span style="color: #B5CEA8;">0.00</span>              <span style="color: #B5CEA8;">1</span>          <span style="color: #B5CEA8;">1</span>               <span style="color: #B5CEA8;">1</span>        <span style="color: #B5CEA8;">101348.88</span>       <span style="color: #B5CEA8;">1</span>
<span style="color: #B5CEA8;">1</span>             <span style="color: #B5CEA8;">2</span>    <span style="color: #B5CEA8;">15647311</span>       <span style="color: #CE9178;">Hill</span>          <span style="color: #B5CEA8;">608</span>   <span style="color: #B5CEA8;">41</span>       <span style="color: #B5CEA8;">1</span>   <span style="color: #B5CEA8;">83807.86</span>              <span style="color: #B5CEA8;">1</span>          <span style="color: #B5CEA8;">0</span>               <span style="color: #B5CEA8;">1</span>        <span style="color: #B5CEA8;">112542.58</span>       <span style="color: #B5CEA8;">0</span>
<span style="color: #B5CEA8;">2</span>             <span style="color: #B5CEA8;">3</span>    <span style="color: #B5CEA8;">15619304</span>       <span style="color: #CE9178;">Onio</span>          <span style="color: #B5CEA8;">502</span>   <span style="color: #B5CEA8;">4...</span></code></pre>
<p style="font-size: 0.9em; color: #666; margin-top: 10px;">
  The target variable for prediction is likely the `Exited` column (binary classification).
</p>

---

<h2 id="modeling-approach" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  ⚙️ Modeling Approach
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This notebook employs a standard approach for building a **Dense Neural Network** for classification:
</p>
<div style="background-color: #F3E5F5; border: 1px solid #AB47BC; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <h3 style="color: #6200EE; font-size: 1.8em; margin-top: 0;">Key Steps:</h3>
  <ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
    <li style="margin-bottom: 10px;"><strong style="color: #6200EE;">Feature Engineering/Preprocessing:</strong> Identification and handling of independent and dependent variables. Categorical features are transformed using One-Hot Encoding, and numerical features are scaled using `StandardScaler` to ensure optimal training.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #6200EE;">Model Initialization:</strong> A `Sequential` Keras model is initialized.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #6200EE;">Adding Layers:</strong> Multiple `Dense` layers are added, forming the core of the neural network. Activation functions (e.g., `relu` for hidden layers, `sigmoid` for the output layer in binary classification) are applied.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #6200EE;">Model Compilation:</strong> The model is compiled with an optimizer (e.g., `adam`), a loss function suitable for binary classification (`binary_crossentropy`), and evaluation metrics (`accuracy`).</li>
    <li style="margin-bottom: 10px;"><strong style="color: #6200EE;">Training:</strong> The model is trained on the preprocessed training data over a specified number of epochs.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #6200EE;">Prediction & Evaluation:</strong> Predictions are made on the test set, and performance is evaluated using a `confusion_matrix`.</li>
  </ul>
</div>

---

<h2 id="features" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  🎯 Features
</h2>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 15px; background-color: #F8BBD0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #C2185B;">🚀 End-to-End ANN Pipeline:</strong> Covers data loading, preprocessing, model building, training, and evaluation.
  </li>
  <li style="margin-bottom: 15px; background-color: #F8BBD0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #C2185B;">📊 Data Preprocessing with Scikit-learn:</strong> Demonstrates `StandardScaler` and `OneHotEncoder` within a `ColumnTransformer`.
  </li>
  <li style="margin-bottom: 15px; background-color: #F8BBD0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #C2185B;">🏗️ Keras Model Building:</strong> Clear implementation of a `Sequential` Dense Neural Network.
  </li>
  <li style="margin-bottom: 15px; background-color: #F8BBD0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #C2185B;">📏 Performance Evaluation:</strong> Utilizes `confusion_matrix` for in-depth model assessment.
  </li>
</ul>

---

<h2 id="prerequisites" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  🛠️ Prerequisites
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  To run this project, ensure you have the following installed:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #6200EE;">Python 3.x</strong></li>
  <li><strong style="color: #6200EE;">Jupyter Notebook</strong> (or JupyterLab, Google Colab)</li>
  <li>Required Libraries:
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install pandas numpy scikit-learn tensorflow keras</code></pre>
  </li>
</ul>

---

<h2 id="how-to-run" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  📋 How to Run
</h2>
<ol style="list-style-type: decimal; padding-left: 20px; font-size: 1.1em; color: #444; line-height: 1.8;">
  <li style="margin-bottom: 10px;">
    <strong style="color: #6200EE;">Download the Notebook:</strong>
    <p style="margin-top: 5px;">Download <code>Artificial_Neural_Network.ipynb</code> from this repository.</p>
    <p style="margin-top: 5px;">Alternatively, open it directly in <a href="https://colab.research.google.com/" style="color: #6200EE; text-decoration: none;">Google Colab</a> for a zero-setup experience.</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #6200EE;">Prepare Data:</strong>
    <p style="margin-top: 5px;">Ensure you have your dataset (e.g., `Churn_Modelling.csv` or similar) in the same directory as the notebook, or adjust the file path in the `pd.read_csv()` call.</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #6200EE;">Install Dependencies:</strong>
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install pandas numpy scikit-learn tensorflow keras</code></pre>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #6200EE;">Run the Notebook:</strong>
    <p style="margin-top: 5px;">Open <code>Artificial_Neural_Network.ipynb</code> in Jupyter or Colab.</p>
    <p style="margin-top: 5px;">Execute each cell sequentially to see the ANN in action!</p>
  </li>
</ol>

---

<h2 id="example-output" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  📈 Example Output
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  During execution, you will see output related to model training progress and final evaluation.
</p>
<h3 style="color: #6200EE; font-size: 1.8em; margin-top: 25px;">Model Training Output (Epochs):</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #CE9178;">Epoch 1/5</span>
<span style="color: #B5CEA8;">1875/1875</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">5s</span> <span style="color: #B5CEA8;">2ms/step - accuracy: 0.8579 - loss: 0.4969</span>
<span style="color: #CE9178;">Epoch 2/5</span>
<span style="color: #B5CEA8;">1875/1875</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">6s</span> <span style="color: #B5CEA8;">3ms/step - accuracy: 0.9531 - loss: 0.1634</span>
<span style="color: #CE9178;">Epoch 3/5</span>
<span style="color: #B5CEA8;">1875/1875</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">9s</span> <span style="color: #B5CEA8;">4ms/step - accuracy: 0.9602 - loss: 0.1210</span>
<span style="color: #CE9178;">...</span></code></pre>

<h3 style="color: #6200EE; font-size: 1.8em; margin-top: 25px;">Confusion Matrix Example:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #CE9178;">[[TN FP]</span>
 <span style="color: #CE9178;">[FN TP]]</span> <span style="color: #6A9955;"># Illustrative example of a confusion matrix structure</span>
<span style="color: #B5CEA8;">array([[1540,   55],</span>
       <span style="color: #B5CEA8;">[ 250,  155]])</span> <span style="color: #6A9955;"># Example actual output values</span></code></pre>
<p style="font-size: 0.9em; color: #666; margin-top: 10px;">
  The confusion matrix provides a detailed breakdown of correct and incorrect classifications.
</p>

---

<h2 id="code-breakdown" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  🧠 Code Breakdown
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Key parts of the notebook's code structure:
</p>

<h3 style="color: #6200EE; font-size: 1.8em; margin-top: 25px;">Data Loading & Preprocessing:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">pandas</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">pd</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.compose</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">ColumnTransformer</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.preprocessing</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">StandardScaler</span>, <span style="color: #9CDCFE;">OneHotEncoder</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.model_selection</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">train_test_split</span>
<span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">numpy</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">np</span>

<span style="color: #9CDCFE;">df</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">pd.read_csv</span>(<span style="color: #CE9178;">'Churn_Modelling.csv'</span>) <span style="color: #6A9955;"># Example dataset name</span>
<span style="color: #9CDCFE;">X</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">df.iloc</span>[:, <span style="color: #B5CEA8;">3</span>:<span style="color: #CE9178;">-1</span>].<span style="color: #9CDCFE;">values</span>
<span style="color: #9CDCFE;">y</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">df.iloc</span>[:, <span style="color: #CE9178;">-1</span>].<span style="color: #9CDCFE;">values</span>

<span style="color: #9CDCFE;">ct</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">ColumnTransformer</span>(<span style="color: #9CDCFE;">transformers</span><span style="color: #CE9178;">=</span>[(<span style="color: #CE9178;">'encoder'</span>, <span style="color: #9CDCFE;">OneHotEncoder</span>(), [<span style="color: #B5CEA8;">1</span>,<span style="color: #B5CEA8;">2</span>])], <span style="color: #9CDCFE;">remainder</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'passthrough'</span>)
<span style="color: #9CDCFE;">X</span> <span style="color: #CE9178;">=</span> <span style="color: #569CD6;">np.array</span>(<span style="color: #9CDCFE;">ct.fit_transform</span>(<span style="color: #9CDCFE;">X</span>))

<span style="color: #9CDCFE;">X_train</span>, <span style="color: #9CDCFE;">X_test</span>, <span style="color: #9CDCFE;">y_train</span>, <span style="color: #9CDCFE;">y_test</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">train_test_split</span>(<span style="color: #9CDCFE;">X</span>, <span style="color: #9CDCFE;">y</span>, <span style="color: #9CDCFE;">test_size</span> <span style="color: #CE9178;">=</span> <span style="color: #B5CEA8;">0.2</span>, <span style="color: #9CDCFE;">random_state</span> <span style="color: #CE9178;">=</span> <span style="color: #B5CEA8;">0</span>)

<span style="color: #9CDCFE;">sc</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">StandardScaler</span>()
<span style="color: #9CDCFE;">X_train</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">sc.fit_transform</span>(<span style="color: #9CDCFE;">X_train</span>)
<span style="color: #9CDCFE;">X_test</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">sc.transform</span>(<span style="color: #9CDCFE;">X_test</span>)</code></pre>

<h3 style="color: #6200EE; font-size: 1.8em; margin-top: 25px;">Building the ANN:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">tensorflow</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">keras</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">keras</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">Sequential</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">keras.layers</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">Dense</span>

<span style="color: #9CDCFE;">classifier</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">Sequential</span>()
<span style="color: #9CDCFE;">classifier.add</span>(<span style="color: #9CDCFE;">Dense</span>(<span style="color: #9CDCFE;">units</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">6</span>, <span style="color: #9CDCFE;">activation</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'relu'</span>, <span style="color: #9CDCFE;">input_dim</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">12</span>)) <span style="color: #6A9955;"># Example input_dim</span>
<span style="color: #9CDCFE;">classifier.add</span>(<span style="color: #9CDCFE;">Dense</span>(<span style="color: #9CDCFE;">units</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">6</span>, <span style="color: #9CDCFE;">activation</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'relu'</span>))
<span style="color: #9CDCFE;">classifier.add</span>(<span style="color: #9CDCFE;">Dense</span>(<span style="color: #9CDCFE;">units</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">1</span>, <span style="color: #9CDCFE;">activation</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'sigmoid'</span>))

<span style="color: #9CDCFE;">classifier.compile</span>(<span style="color: #9CDCFE;">optimizer</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'adam'</span>, <span style="color: #9CDCFE;">loss</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'binary_crossentropy'</span>, <span style="color: #9CDCFE;">metrics</span><span style="color: #CE9178;">=</span>[<span style="color: #CE9178;">'accuracy'</span>])</code></pre>

<h3 style="color: #6200EE; font-size: 1.8em; margin-top: 25px;">Training and Evaluation:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #9CDCFE;">classifier.fit</span>(<span style="color: #9CDCFE;">X_train</span>, <span style="color: #9CDCFE;">y_train</span>, <span style="color: #9CDCFE;">batch_size</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">32</span>, <span style="color: #9CDCFE;">epochs</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">5</span>) <span style="color: #6A9955;"># Example epochs</span>

<span style="color: #9CDCFE;">y_pred</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">classifier.predict</span>(<span style="color: #9CDCFE;">X_test</span>)
<span style="color: #9CDCFE;">y_pred</span> <span style="color: #CE9178;">=</span> (<span style="color: #9CDCFE;">y_pred</span> <span style="color: #CE9178;">></span> <span style="color: #B5CEA8;">0.5</span>)

<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.metrics</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">confusion_matrix</span>
<span style="color: #9CDCFE;">cm</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">confusion_matrix</span>(<span style="color: #9CDCFE;">y_test</span>, <span style="color: #9CDCFE;">y_pred</span>)
<span style="color: #569CD6;">print</span>(<span style="color: #9CDCFE;">cm</span>)</code></pre>

---

<h2 id="contribute" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #BB86FC; padding-bottom: 10px;">
  🤝 Contribute
</h2>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Contributions are highly encouraged! Whether you'd like to improve model performance, experiment with different activation functions or optimizers, add more detailed visualizations (e.g., loss/accuracy curves), or refine the data preprocessing steps, feel free to open an issue or submit a pull request. Let's make this a robust learning resource for ANNs! 🌟
</p>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 15px auto 0; line-height: 1.6;">
  Star this repo if you find it helpful! ⭐
</p>
<p align="center" style="font-size: 1em; color: #777; margin-top: 30px;">
  Created with 💖 by Chirag
</p>
