*A Report on*  
**A Comprehensive Study of  
Machine Learning Algorithms  
on Forest Cover Dataset**  

*Presented by*  

<table>
<tbody>
<tr>
<td style="text-align: left;"><strong>Shreya Gupta</strong></td>
<td style="text-align: left;">MT2025724</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Anirudh Sharma</strong></td>
<td style="text-align: left;">MT2025732</td>
</tr>
</tbody>
</table>

*Under the Guidance of*  
**Prof. Sushree Behera**  
**Machine Learning**  

International Institute of Information Technology, Bangalore  

2026-03-17

# Abstract

In this project, two different classification datasets were utilized:
one for predicting whether a person is a smoker based on biosignals, and
another for identifying forest cover types from environmental features.
The primary objective was the application of all machine learning models
covered in the curriculum, including Logistic Regression, SVM, Neural
Networks, K-Means, Agglomerative Clustering, DBSCAN, and Gaussian
Mixture Models.

For both datasets, data preprocessing and exploratory data analysis were
initially performed to assess the data structure and the relationships
between different features. Subsequently, each model was trained
separately, and its performance was evaluated using appropriate metrics
for binary and multiclass classification. Finally, all models were
compared to identify the most effective ones, and the factors
contributing to their performance were investigated.

Overall, insight was gained regarding the behavior of different
algorithms across various dataset types, the significance of
preprocessing, and how model performance is influenced by data
complexity.

[<u><span style="color: blue">https://github.com/Minecraftian14/ml-cmp-forest-cover</span></u>](https://github.com/Minecraftian14/ml-cmp-forest-cover)

# Introduction

Forests play a crucial role in maintaining ecological balance,
supporting biodiversity, and influencing global climate patterns.
Understanding forest composition is important for conservation, land
management, wildfire prevention, and ecological research. One of the key
aspects of forest analysis is determining the type of forest cover
present in a given region. Forest cover types are influenced by a mix of
environmental factors such as soil properties, elevation, climate, and
topographic conditions. These features interact in complex ways, and
even small variations can result in shifts between different types of
vegetation. Because of this complexity, accurately identifying forest
cover types is a challenging but essential task for researchers,
ecologists, and environmental agencies.

Traditional methods of determining forest cover-such as manual field
surveys or rule-based ecological models-tend to rely on predefined
thresholds or human expertise. While useful, these methods often fail to
capture nonlinear relationships and subtle interactions in environmental
data. In contrast, modern machine learning (ML) techniques provide a
powerful way to analyze large-scale ecological datasets and detect
patterns that may not be obvious using classical approaches. ML
algorithms can learn from multidimensional environmental features and
produce accurate predictions about forest categories, making them highly
valuable for applications in land-use planning, habitat conservation,
and environmental monitoring.

Machine learning has already shown strong potential in environmental
science, particularly in areas like crop classification, remote sensing
analysis, wildlife detection, and climate modeling. ML models can
process large datasets collected from sensors, satellites, and field
measurements to reveal hidden structures in the data. For forest
ecosystems, where multiple factors simultaneously influence vegetation
growth, ML provides a systematic way to classify cover types and detect
environmental trends. With forests facing increasing pressures from
climate change, wildfires, and human activity, data-driven systems for
monitoring cover types can help decision-makers plan better
interventions and manage resources efficiently.

The motivation for this project comes from the need to build a reliable
prediction system that can classify forest cover types based on
environmental attributes. The dataset used in this study includes
quantitative measurements of soil characteristics, elevation, slope,
aspect, distance to hydrology and roadways, and wilderness areas, all of
which influence forest composition. Since the dataset is purely
numerical and relatively large, it offers a good opportunity to explore
different preprocessing steps and modeling strategies. The first phase
involved conducting an Exploratory Data Analysis (EDA) to understand
data distributions, correlations, and feature importance. Visualizations
such as histograms, box plots, and basic scatter-plots helped identify
patterns and detect potential issues such as skewed values.

Preprocessing steps such as scaling numerical features and checking for
outliers were performed to prepare the data for model training. Because
this is a multiclass classification problem involving seven forest cover
categories, choosing the right evaluation metrics and handling class
imbalance were important considerations. Standardization was applied to
numerical features to ensure that models such as logistic regression,
SVM, and neural networks could perform optimally. Since there are no
categorical features in this dataset, encoding methods were not
required.

The goal of this project was to build and compare multiple machine
learning models using the algorithms taught in class. These included
Logistic Regression, Support Vector Machines, Neural Networks, K-Means,
Agglomerative Clustering, DBSCAN, and Gaussian Mixture Models. Each
model was trained individually, and their performances were evaluated
using appropriate accuracy metrics for multiclass classification.
Hyperparameters were tuned to ensure that every model was tested fairly.
Cluster-based models were also included for unsupervised analysis,
allowing us to explore how well the data naturally groups into cover
types without labels.

The dataset was split into training and testing sets to evaluate how
well each model generalizes to unseen data. After training all models,
their results were compared, and the strengths and weaknesses of each
approach were analyzed. Some models performed better because they
handled high-dimensional, continuous data more effectively, while others
struggled due to the complexity of the classification task. The
comparison helped illustrate how different algorithms behave on
real-world environmental datasets.

Beyond model accuracy, the broader importance of this project lies in
its practical applications. Accurate forest cover classification can
support environmental conservation, wildfire planning, sustainable
forestry, and ecosystem research. By developing data-driven models, we
can help automate the monitoring process and contribute to a better
understanding of ecological patterns. Machine learning can thus serve as
a useful tool for environmental scientists, enabling large-scale
analysis that would be difficult to achieve manually.

# Methodology

A typical machine learning workflow usually moves step-by-step: we load
the data, explore it, clean and preprocess it, train different models,
and then produce the final predictions or outputs.

But in this project, instead of following this straight, one-directional
path, I used a more flexible and cycle-based approach inspired by how
real data science pipelines are built.

One of the main changes is how visualizations are used. Rather than
treating plots and graphs as something done only at the beginning during
EDA, I included visualization throughout the entire process. This means
I kept checking the data during preprocessing, during feature
transformations, and even while evaluating models. Doing this made it
easier to catch unusual patterns early and helped me understand how each
model responds to different features.

Another important part of the methodology is modularity. Since many of
the models require similar preprocessing steps or repeated experiment
setups, it made sense to organize the code using helper functions and
reusable components. This not only reduced repetition but also made the
experiments more structured and easier to compare. Having these small
utility methods allowed me to run tests more smoothly and keep the
workflow clean and consistent across different models.

## Dataset Description

The Forest Cover Type dataset used in this project is provided in a
clean and ready-to-use CSV format, which makes the loading process very
simple. Since the file already contains structured numerical features
with no complicated formatting issues, we can import it directly into a
DataFrame without requiring any special parsing. For this project, the
dataset is loaded into a variable named `ds_source`.

Unlike some Kaggle competitions, this project does not involve
generating submission files, so there is no separate `ds_test` dataset.
All analysis, preprocessing, and model training are performed on the
single dataset provided.

After loading the data, one of the first steps is to rename several
columns. Many of the original feature names are long, especially the
distance-related and hillshade features, which makes them harder to read
in tables and nearly impossible to display neatly in plots. To improve
visualization quality and keep the graph labels clean, the following
renaming scheme is applied:

- Elevation =&gt; Elev

- Aspect =&gt; Aspc

- Slope =&gt; Slpe

- Horizontal\_Distance\_To\_Hydrology =&gt; HDTH

- Vertical\_Distance\_To\_Hydrology =&gt; VDTH

- Horizontal\_Distance\_To\_Roadways =&gt; HDTR

- Hillshade\_9am =&gt; H09A

- Hillshade\_Noon =&gt; H12P

- Hillshade\_3pm =&gt; H03P

- Horizontal\_Distance\_To\_Fire\_Points =&gt; HDFP

- Wilderness\_AreaN =&gt; WAN

- Soil\_TypeN =&gt; STNN

These shorter names make plots cleaner, tables more compact, and the
overall analysis easier to follow. While the change is small, it
improves consistency across visualizations and helps maintain
readability throughout the project.

## Exploratory Data Analysis (EDA)

The goal of the Exploratory Data Analysis stage is to carefully examine
the dataset for any missing values, unusual patterns, extreme
observations, or other statistical irregularities. Even though this
stage may look routine, it serves a much deeper purpose than simply
producing charts-it helps reveal important characteristics of the data
that directly affect how preprocessing and modeling decisions are made.

To build a solid understanding of the dataset, multiple visualizations
and descriptive statistics are used. These include distribution plots,
correlation checks, and feature comparisons. Each plot is chosen
deliberately so that it adds something meaningful to our interpretation,
rather than being included just for decoration. By analyzing these
visual patterns and summary statistics, we gain clearer insights into
how the features behave, which ultimately guides how we clean the data
and prepare it for the machine learning models.

## Data Preprocessing & Feature Engineering

The data preprocessing and feature engineering stage acts as the
foundation of the entire machine learning pipeline. This is where the
raw dataset is converted into a clean, structured, and meaningful form
that models can understand and learn from. Every transformation applied
here is based directly on insights collected during the EDA phase,
ensuring that the preprocessing choices are data-driven and not
arbitrary.

### Transformation Strategy

The main goal of this stage is to improve both the interpretability and
predictive value of the features while keeping the dataset compatible
with different types of models. Since linear models, neural networks,
and tree-based algorithms each have different expectations about input
data, the preprocessing strategy is designed to work well across all of
them.

**Numerical Transformations** Most numerical features in the dataset
(except *Aspc*) showed a skewed Gaussian-like distribution. To correct
this and make the features more symmetric, we apply a PowerTransformer.
PowerTransformer uses the Yeo-Johnson transformation, which is similar
to Box-Cox but works even when the data contains zeros or negative
values. The goal is to stabilize variance and make the distribution
closer to normal, which helps many models learn more effectively.

**Categorical Transformations** The dataset already comes with its
categorical attributes (Wilderness Areas and Soil Types) fully one-hot
encoded, so no additional encoding steps are needed. This is convenient
because it ensures immediate compatibility with all models.

### Scaling and Standardization

To prevent features with larger magnitudes from dominating others,
normalization is essential. After applying the PowerTransformer, the
numerical features are already scaled to behave like standardized
variables.

Meanwhile, the categorical one-hot encoded columns naturally lie in the
0-1 range, so no further scaling is necessary. This creates a balanced
feature space suitable for both linear and non-linear models.

## Setups and Helpers

Although this section does not directly appear in the final report
output, it plays a foundational role within the notebook implementation.
Executed immediately after the import statements, it handles several
preliminary configurations essential for smooth experimentation. These
include:

- Initialization of the notification and logging system

- PyPlot and visualization styling adjustments

- Random seed management for reproducibility

- Notebook display and formatting controls

- Model training, saving, and submission automation

- Definition and registration of POP operations

## Model Training and Evaluation

The final step of this stage involves constructing the models, tuning
their hyperparameters, and evaluating their performance. A consistent
training framework is used across all algorithms, but with room for
model-specific adjustments when needed. This balance allows fair
comparison while still allowing each model to perform at its best.

### Data Splitting

Before model training, the fully preprocessed dataset is divided into
training (96

### Model Selection and Training

A diverse suite of models was implemented to evaluate various learning
paradigms and understand how different algorithms behave on the Forest
Cover Type dataset. Each model provides a unique perspective on the data
and contributes to the overall comparative study:

- Logistic Regression - Served as the baseline linear classifier. It was
  tested with different regularization strategies (L1, L2) and solver
  options to assess how well linear decision boundaries can separate the
  classes.

- Support Vector Machine (SVM) - Explored both linear and non-linear
  kernels to capture more complex decision boundaries. Despite being
  computationally heavier, SVM helps evaluate margin-based
  classification behavior.

- Neural Networks - Implemented as a multi-layer perceptron to model
  non-linear relationships. It provides insight into how well deep
  feature interactions can improve predictive power.

- K-Means Clustering - Used in an unsupervised setting to examine
  natural grouping patterns within the dataset. This helps analyze
  whether forest cover types form well-separated clusters in feature
  space.

- Agglomerative Clustering - Offers a hierarchical perspective, allowing
  us to observe how clusters merge progressively and whether any
  meaningful structure emerges.

- DBSCAN - Helps detect density-based patterns and potential anomalies.
  It is particularly useful for understanding irregular or noise-prone
  regions in the data.

- Gaussian Mixture Models (GMM) - Evaluates whether probabilistic soft
  clustering aligns with the actual class distribution and highlights
  overlapping feature regions.

### Hyperparameter Optimization

To ensure that each model performs at its best, Grid Search was used to
systematically explore multiple combinations of hyperparameters. Grid
Search exhaustively evaluates every possible parameter setting defined
in the search space.

This is combined with Cross-Validation, where the training set is split
into multiple folds. Each model is trained on a subset of the folds and
validated on the remaining one, cycling through all folds.

This process ensures that the chosen hyperparameters generalize well and
are not overfitted to a single train-validation split. It also improves
the reliability and fairness of comparisons across models.

### Evaluation Metrics

Since this is a multiclass classification problem, a single score is not
enough to judge performance. Therefore, several metrics were used to
provide a complete picture:

- Precision – Measures how many of the predicted samples for a class are
  actually correct.

- Recall – Measures how many actual samples of a class were successfully
  identified.

- F1-Score – The harmonic mean of precision and recall, useful when
  dealing with class imbalance or uneven performance across classes.

- Confusion Matrix – Provides a detailed breakdown of correct and
  incorrect predictions for every class, helping visualize where models
  struggle.

These metrics allow fine-grained evaluation and make it easier to
compare models not just by accuracy but by their behavior on individual
forest cover types.

# Dataset Description

The Forest Cover Type dataset consists of tree observations collected
from four major regions of the Roosevelt National Forest in Colorado.
Each observation represents a 30 × 30 meter plot of land, and with over
half a million samples, the dataset provides a rich and detailed view of
forest landscapes. All the features in this dataset are cartographic
variables, meaning they come from ground-based measurements rather than
satellite or remote-sensing data.

The dataset captures a variety of environmental characteristics such as
elevation, slope, aspect, soil type, distance to hydrology or roadways,
hillshade values, and other topographical factors. It also includes
indicators for wilderness areas and soil classifications, both
represented using binary one-hot encoded columns. These variables help
describe how terrain and environmental conditions influence the type of
forest vegetation present.

The task is to predict the forest cover type for each 30 × 30 meter
plot. The ground-truth labels were provided by the US Forest Service
(USFS) Region 2 Resource Information System (RIS), which contains
expert-verified vegetation data. Most of the independent variables
originate from the US Geological Survey (USGS) and USFS resources. All
features are provided in their raw numerical form, and no scaling or
normalization is applied beforehand.

This combination of physical terrain measurements and vegetation labels
makes the dataset ideal for exploring how environmental factors
contribute to forest composition, and it provides a strong foundation
for testing a variety of machine learning models.

The features are grouped as follows:

1.  **Topographical / Terrain Features**

    - **Elevation** Height of the plot above sea level (in meters).
      Different trees grow at different altitudes due to temperature and
      oxygen differences.

    - **Aspect** Direction the slope faces (0–360°). Sunlight exposure
      affects soil moisture and vegetation patterns.

    - **Slope** Steepness of the terrain (in degrees). Steeper areas may
      limit tree growth or influence water runoff.

2.  **Distance-Based Features**

    - **Horizontal\_Distance\_To\_Hydrology** Horizontal distance to the
      nearest water source (streams, lakes). Trees differ in their water
      requirements.

    - **Vertical\_Distance\_To\_Hydrology** Elevation difference
      relative to nearby water bodies. Lower/sunken areas may hold more
      moisture.

    - **Horizontal\_Distance\_To\_Roadways** Distance to the nearest
      road. Roads can alter soil, introduce human disturbance, or
      influence forest composition.

    - **Horizontal\_Distance\_To\_Fire\_Points** Distance to known
      wildfire ignition points. Fire-tolerant species may dominate
      closer regions.

3.  **Hillshade Features**

    - **Hillshade\_9am** Amount of sunlight at 9 AM, given terrain
      orientation.

    - **Hillshade\_Noon** Sunlight at solar noon, usually the brightest
      time.

    - **Hillshade\_3pm** Sunlight at 3 PM.

4.  **Wilderness Area Indicators** The dataset includes four wilderness
    areas, encoded as: Wilderness\_Area1, Wilderness\_Area2,
    Wilderness\_Area3, Wilderness\_Area4. Each is a binary (0/1) feature
    that indicates whether a plot belongs to that specific protected
    area. Different wilderness regions have different regulations,
    climates, soil conditions, and vegetation patterns-which directly
    influence forest type.

5.  **Soil Type Indicators** There are 40 soil types, represented as:
    Soil\_Type1, Soil\_Type2, …, Soil\_Type40. Each is a binary feature
    indicating the presence of a specific soil classification. Soil
    characteristics such as composition, drainage, pH, and nutrient
    availability strongly affect which tree species can grow.

# Exploratory Data Analysis

This section presents a detailed exploratory data analysis of the
dataset. We examine data completeness, outlier presence, feature
distributions, and sensitivity to extreme values. Each subsection
summarizes visual analyses and their corresponding insights.

## Missing Value Detection

<figure id="plot:eda_missing_counts" data-latex-placement="ht">
<img src="../attachments/figures/plot_Missing Value Counts.png"
style="width:50.0%" />
<figcaption>Missing Value Counts</figcaption>
</figure>

**Observation:** There is not a single missing value in any of the
columns.  
**Inference:** The dataset is well-structured and complete. No
imputation, removal, or flagging of missing values is required.

## Outlier Detection

<figure id="plot:eda_outlier_counts" data-latex-placement="ht">
<img src="../attachments/figures/plot_Outlier Counts.png"
style="width:50.0%" />
<figcaption>Outlier Counts</figcaption>
</figure>

**Observation:** *VDTH* and a few others exhibit a small number of
values beyond the interquartile range (IQR).  
**Inference:** Since the dataset is beyond 500 k samples, a small number
of outliers such as 173 wont matter.

## Duplication Detection

**Observation:** There is not even a single duplicate row in the
dataset.  
**Inference:** The dataset is clean in this regard, and there is no need
to drop any of the rows.

## Feature Distribution (Numerical)

<figure id="plot:feature_distribution_numerical"
data-latex-placement="ht">
<img
src="../attachments/figures/plot_Feature Distribution (Numerical).png"
style="width:50.0%" />
<figcaption>Numerical Features Distribution</figcaption>
</figure>

**Observation:**

- **Simplest Gaussians:** *Elev, Slpe, VDTH, H03P*

- **Skewed Gaussians:** *HDTH, H09A, H12P, HDFP*

- **Not Gaussians:** *Aspc, HDTR*

**Inference:** In either case, they can do good with some level of power
transformations.

## Feature Distribution (Wilderness)

<figure id="plot:feature_distribution_wilderness"
data-latex-placement="ht">
<img
src="../attachments/figures/plot_Feature Distribution (Wilderness).png"
style="width:50.0%" />
<figcaption>Wilderness Features Distribution</figcaption>
</figure>

**Observation:**

- *Wilderness Area 1 and 3* is well-balanced.

- *Wilderness Area 2 and 4* are highly skewed.

**Inference:** Wilderness Area 2 and 4 still have a std of 0.2, which is
still nearly half of what Wilderness Area 1. No need to process this
feature further.

## Feature Distribution (Soil Type)

<figure id="plot:feature_distribution_soil_type"
data-latex-placement="ht">
<img
src="../attachments/figures/plot_Feature Distribution (Soil Type).png"
style="width:50.0%" />
<figcaption>Soil Type Features Distribution</figcaption>
</figure>

**Observation:** Every soil type is pretty well distributed.
**Inference:** No actions needed.

## Feature Spread (Numerical)

<figure id="plot:feature_spread_numerical" data-latex-placement="ht">
<img src="../attachments/figures/plot_Feature Spread (Numerical).png"
style="width:50.0%" />
<figcaption>Numerical Feature Spread</figcaption>
</figure>

**Observation:**

1.  **Differing Scales**

    - *Slpe* has a small scale (1e0:1)

    - *Hillshade features* have medium scales (1e0:2)

    - *HDTR, and HDFP* have large scales (1e0:3)

2.  **Skewness**

    - *Slpe, HDTH, HDTR, and HDFP* all show a concentration of data at
      the lower end.

    - *H09A and H12P*are bunched towards the top.

3.  **Outliers**

    - *VDTH* shows a massive cluster of outliers on the upper end.

    - *H12P* There are distinct outliers at the very bottom.

**Inference:**

- It would be a good idea to standardize the features to introduce
  stability and prevent overshadowing of the features with smaller
  scales.

- Will apply power transform to adjust the skewed-ness of the various
  features identified above.

- As verified earlier, the outlier count is not very high; therefore,
  they can be set aside.

## Outlier Sensitivity (Numerical)

<figure id="plot:outlier_sensitivity_numerical"
data-latex-placement="ht">
<img
src="../attachments/figures/plot_Outlier Sensitivity (Numerical).png"
style="width:50.0%" />
<figcaption>Numerical Outlier Sensitivity</figcaption>
</figure>

**Observation:**

- **Extreme Invariance:** *HDTR and HDFP* show exponentially
  differentiated presence compared to other features.

- **Negligible Invariance:** *Slpe and H03P* have no offset to show at
  all, likely to be the most sensitive to outliers.

- In general, all features lie beyond the threshold.

**Inference:** A standard scaler could likely solve all issues.

## Correlation Matrix

<figure id="plot:correlation" data-latex-placement="ht">
<img
src="../attachments/figures/plot_Correlation Matrix Main Dataset.png"
style="width:50.0%" />
<figcaption>Correlation Matrix</figcaption>
</figure>

**Observation:**

1.  Most features show a moderate correlation, between -.5 to +.5.

2.  *H03P, and H09A* show some level of negative correlation, around
    -0.75.

3.  Other notable relations are shown by *Elev and Slpe*, *Aspc and
    HDTR*.

**Inference:** The relationships between the Hillshade features indicate
overlapping information. However, they don’t stand on absolute
indifference either. Since none of the features are strongly related,
let the models create any associations if necessary.

## Principal Components

<figure id="plot:pca" data-latex-placement="ht">
<img src="../attachments/figures/plot_3D PCA Projection.png"
style="width:50.0%" />
<figcaption>Principal Components</figcaption>
</figure>

**Observation:** The PCA projection shows a large degree of overlap
among most classes, with no clearly separated clusters. All the forest
cover categories blend into one another, indicating that their feature
distributions are similar in the reduced 3D space.

**Inference:** This overlap suggests that the variables contributing to
cover type classification are continuous and interrelated, making sharp
boundaries between classes difficult to capture with PCA. It also
implies that while PCA captures overall variance, it may not fully
separate cover type categories-nonlinear methods (e.g., t-SNE or UMAP)
might reveal clearer class distinctions.

## Summary of EDA Findings

The dataset shows complete feature coverage with no missing values and
well-distributed numerical and categorical variables. While a few
numerical features contain outliers, their low frequency means they do
not pose a significant problem for modeling. Overall, the dataset is
clean, consistent, and well-suited for machine learning analysis.

# Model Training and Evaluation

Following the data preprocessing and feature engineering stages, the
next step was to train a set of machine learning models to predict the
forest cover type for each 30×30 meter land segment. The goal was to
explore a wide range of classifiers-spanning simple linear methods to
more complex non-linear and probabilistic approaches-to understand how
different learning paradigms perform on this multi-class, terrain-based
dataset. By evaluating multiple algorithm families, we gain a clearer
picture of which modeling strategies are best suited for the
environmental patterns present in the forest cover data.

## Experimental Setup and Data Splitting

To ensure robust evaluation, the dataset was divided into training and
testing subsets using a stratified split to maintain class balance
across all partitions.

- 96% (557771 samples) of the data was allocated for model development
  (training and validation), and

- 4% (23241) was held out as a final test set for unbiased evaluation.

During model training, Grid Search was paired with K-Fold
Cross-Validation (using k values between 2 and 5) to get more dependable
estimates of model performance, especially during hyperparameter tuning.
This strategy reduces the risk of overfitting and gives a better sense
of how well the model generalizes across different portions of the data.

However, because the dataset is extremely large (around 550 thousand
samples), running an exhaustive grid search becomes computationally
impractical on a local machine. As a result, the search space had to be
carefully narrowed down-focusing only on the most impactful
hyperparameters and limiting the number of combinations explored-to make
the tuning process possible.

## Data Pipelining

A lightweight data pipeline was used to keep the preprocessing steps
consistent and reproducible. Since only the numerical features required
transformation, a ColumnTransformer was created to apply a
PowerTransformer to the Gaussian-like features while leaving the one-hot
encoded wilderness and soil type features unchanged. Using a pipeline
ensures that the same preprocessing applied during training is also
consistently applied to the validation set, keeping the workflow clean
and reliable. As for the remaining features, since one hot encoding has
already been applied to them, we do not have to create any more
transformers.

### Log Transform

The log transform reduces skewness by compressing large values more than
small ones. It is commonly used when data is strictly positive and has a
long right tail. However, it cannot be applied to zero or negative
values.

### Box-Cox Transform

The Box-Cox transformation is a generalization of the log transform. It
can automatically choose an optimal power parameter to make the
distribution more Gaussian-like. It still requires all input values to
be strictly positive.

### Yeo-Johnson Transform

The Yeo-Johnson transformation is similar to Box-Cox but allows zero and
negative values, making it more flexible. It is the version used inside
scikit-learn’s PowerTransformer, and it helps stabilize variance and
normalize skewed numerical features.

## Model Selection Strategy

The project adopted a comparative modeling framework, where diverse
algorithms were evaluated under a consistent preprocessing setup. By
examining multiple learning paradigms-both supervised and
unsupervised-we aimed to understand how different model families
interpret the forest cover data and where each method excels or
struggles.

1.  **Supervised Classification Models** These algorithms directly learn
    to predict the forest cover type from labeled training data.

    - Logistic Regression

    - Support Vector Machine (SVM)

    - Neural Network (MLP Classifier)

2.  **Unsupervised Clustering Models** These models attempt to group the
    data based on structure alone, without using labels. They are
    included to explore natural separations in the feature space.

    - K-Means Clustering

    - Agglomerative (Hierarchical) Clustering

    - DBSCAN

    - Gaussian Mixture Models (GMM)

### Algorithm Overviews

1.  **Logistic Regression** Acts as the baseline linear classifier. It
    models decision boundaries using a linear combination of features
    and supports regularization (L1/L2) to prevent overfitting. Despite
    its simplicity, it provides a strong benchmark for comparison.

    <figure id="plot:Best_LR_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best LR Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on Logistic Regression</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSLR" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSLR.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix Logistic Regression</figcaption>
    </figure>

2.  **Support Vector Machine (SVM)** SVM attempts to find the best
    separating hyperplane by maximizing the margin between classes. With
    different kernels (linear, RBF), it can capture both simple and
    moderately complex decision boundaries.

    <figure id="plot:Best_SV_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best SV Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on Support Vector Machine</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSSV" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSSV.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix Support Vector Machine</figcaption>
    </figure>

3.  **Neural Network (MLP)** A multi-layer perceptron that learns
    non-linear relationships through hidden layers and activation
    functions. Useful for capturing deeper feature interactions that
    linear models might miss.

    <figure id="plot:Best_MP_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best MP Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on Neural Network</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSMP" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSMP.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix Neural Network</figcaption>
    </figure>

4.  **K-Means Clustering** Partitions of the dataset into k clusters by
    minimizing within-cluster variance. Helpful for exploring whether
    forest types form distinct geometric groups.

    <figure id="plot:Best_KM_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best KM Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on K-Means Clustering</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSKM" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSKM.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix K-Means Clustering</figcaption>
    </figure>

5.  **Agglomerative Clustering** A hierarchical, bottom-up clustering
    method that repeatedly merges the closest clusters. It provides
    structural insight into how data points group together at different
    similarity thresholds.

    <figure id="plot:Best_AC_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best AC Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on Agglomerative Clustering</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSAC" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSAC.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix Agglomerative Clustering</figcaption>
    </figure>

6.  **DBSCAN** A density-based algorithm that groups together points in
    high-density regions while marking isolated points as noise. Useful
    for identifying irregular or non-spherical clusters.

    <figure id="plot:Best_DS_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best DS Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on DBSCAN</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSDS" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSDS.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix DBSCAN</figcaption>
    </figure>

7.  **Gaussian Mixture Models (GMM)** A probabilistic clustering method
    assuming data is generated from overlapping Gaussian distributions.
    Unlike K-means, it supports soft clustering and can capture complex
    cluster shapes.

    <figure id="plot:Best_GM_Config" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Best GM Config.png"
    style="width:50.0%" />
    <figcaption>Grid Search on Gaussian Mixture Models</figcaption>
    </figure>

    <figure id="plot:ConfusionMatrix_GSGM" data-latex-placement="ht">
    <img src="../attachments/figures/plot_Confusion Matrix_ GSGM.png"
    style="width:50.0%" />
    <figcaption>Confusion Matrix Gaussian Mixture Models</figcaption>
    </figure>

## Hyperparameter Optimization

Hyperparameter tuning played a critical role in enhancing model
performance.

- Regularization strength (for logistic regression),

- *Logistic Regression:* Regularization Type (penalty) and Factor (C),
  Maximum number of iterations.

- *Support Vector Machine:* Kernel, Degree, Maximum number of
  iterations.

- *Neural Networks:* Hidden Layer Count and Sizes, Maximum number of
  iterations.

- *K-Means:* Number of Clusters, Maximum number of iterations.

- *Agglomerative Clustering:* No hyperparameters (Number of final
  clusters formed is kept the same as the number of targets).

- *Gaussian Mixture Modeling:* Maximum number of iterations.

- *DBScan:* Epsilon, Min Samples.

## Model Evaluation Metrics

Model performance was primarily evaluated using Accuracy, but since the
dataset is not perfectly balanced, relying on accuracy alone would not
give a complete picture. To address this, additional metrics such as
Precision, Recall, and F1-Score were monitored during cross-validation
to better understand class-wise performance and identify which forest
cover types were harder for the models to distinguish. The evaluation
setup provided:

- Cross-validation scores (mean ± standard deviation),

- Final performance on the stratified test split,

- Detailed metric summaries for comparing different model
  configurations.

To interpret prediction patterns more clearly, confusion matrices were
generated, allowing us to inspect misclassifications between specific
cover types. These visualizations were particularly helpful for
understanding overlap among similar vegetation categories.

## Comparative Analysis and Observations

The performance comparison between supervised and unsupervised models
revealed clear distinctions in how well each category handled the forest
cover classification task.

- **Supervised Models (LR, SVM, NN):**  
  Since supervised models learn directly from labeled examples, they
  consistently achieved higher accuracy and more stable predictions
  across all cross-validation folds. Overall, supervised models had a
  significant advantage because they explicitly use the ground-truth
  cover type labels during training, allowing them to learn
  class-specific patterns that unsupervised algorithms cannot access.

- **Unsupervised Models (KM, AC, DS, GM):**  
  Unsupervised models, by design, operate without label information, and
  therefore their performance was noticeably weaker for the actual
  classification task. Instead of predicting forest cover types, they
  attempt to group samples based solely on feature similarities. These
  results are expected: without access to the true forest cover labels,
  unsupervised models have no way to learn class boundaries, making them
  unsuitable as primary predictors.

# Results and Discussion

This chapter summarizes how the different models performed on the forest
cover dataset, how preprocessing influenced the results, and what
insights were gained from observing the behavior of both supervised and
unsupervised algorithms.

## Quantitative Results

All supervised models were evaluated using stratified 2 to 5 fold
cross-validation, followed by testing on the held-out test split.
Metrics such as accuracy, precision, recall, and F1-score were averaged
across folds for consistency. In general:

- Neural Networks achieved the highest accuracy among supervised models.

- Logistic Regression lagged behind because the feature–target
  relationships were largely non-linear.

- SVM performed was significantly slower due to dataset size.

- Unsupervised models showed noticeably poorer alignment with the true
  cover type labels, which is expected since they do not use label
  information.

A simple ranking based on accuracy placed Neural Networks first,
followed by Logistic Regression, then Support Vector Machine. Clustering
models did not produce competitive class-label predictions but were
still useful for exploring natural group structures in the data.

Confusion matrices revealed that certain forest types were harder to
separate, particularly those with similar elevations or soil
compositions.

<figure id="plot:model_accuracy_comparison" data-latex-placement="ht">
<img src="../attachments/figures/plot_Model Accuracy Comparison.png"
style="width:50.0%" />
<figcaption>Model Accuracy Comparison</figcaption>
</figure>

## Impact of Preprocessing and Feature Engineering

### Feature Transformations

PowerTransformer significantly improved the symmetry of numerical
features. This benefited linear models and SVM the most, where accuracy
increased by a noticeable margin. Neural networks were less sensitive
but still showed more stable training behavior after transformation.

### Outlier Handling

Outliers were retained because they seemed to reflect natural variation
in terrain measurements. Removing them did not improve performance and
occasionally worsened generalization.

### Class Imbalance

Although some classes had fewer samples, stratified splitting helped
preserve their proportional presence during training. Since no
oversampling was applied, metrics like recall and F1-score helped
highlight classes that were harder to detect.

### Feature Encoding

The dataset already arrived with one-hot encoded wilderness and soil
type features. This worked well for all supervised models and avoided
unnecessary preprocessing overhead.

## Model Behavior and Interpretations

### Linear Models (Logistic Regression)

Logistic Regression was fast and easy to interpret but struggled due to
the strong non-linearity in the dataset. Terrain and soil features
interact in complex ways that simple linear boundaries cannot capture.

### Kernel-Based Models (SVM)

SVM could not handle the dataset any better, might require higher
degrees. Unfortunately, due to computational limitations, that’s not
possible.

### Neural Network

The neural network performed the best, likely because it can learn
multiple levels of non-linear interactions across the cartographic
features. It balanced accuracy and generalization reasonably well.

### Unsupervised Models

Clustering algorithms (K-Means, Agglomerative, DBSCAN, GMM) were unable
to match the performance of supervised models. Since they receive no
label information, their cluster assignments did not align well with the
actual forest cover categories. They were more useful for understanding
high-level structure rather than classification.

<figure id="plot:metric_comparison" data-latex-placement="ht">
<img
src="../attachments/figures/plot_Metric Comparison over Top Models.png"
style="width:50.0%" />
<figcaption>Metric Comparison over Top Models</figcaption>
</figure>

## Evaluation Metrics Beyond Accuracy

Accuracy alone did not capture the full story due to class imbalance.
Therefore:

- Precision and Recall revealed that minority classes were consistently
  harder to predict.

- F1-Score helped identify trade-offs between false positives and false
  negatives.

- Confusion matrices made misclassification patterns clear, especially
  for the forest types that shared similar soil or elevation values.

These additional metrics provided a more realistic view of how the
models performed across all seven classes.

## Discussion of Findings

### Non-linearity in Terrain Data

The stronger performance of Neural Networks indicates that forest cover
classification depends heavily on non-linear interactions among
elevation, soil type, and distance-based features.

### Outlier Retention

Keeping outliers likely helped the models generalize better, since these
values reflect natural landscape variation rather than noise.

### Unsupervised vs. Supervised Models

Clustering methods struggled not because they performed poorly, but
because they were simply not designed to leverage label information.
This highlights the importance of supervised learning for structured
prediction tasks like this one.

### Efficiency vs. Accuracy

Neural Networks offered the best accuracy but required more training
time. Logistic Regression was extremely fast but significantly less
accurate. SVM balanced both but suffered from scalability issues.

## Limitations

- Dataset imbalance affected recall in certain classes.

- No cost-sensitive evaluation was applied; some misclassifications may
  be more harmful in real ecological analysis.

- Neural networks remain less interpretable, making it harder to explain
  decisions directly.

- Clustering models are fundamentally misaligned with the classification
  goal, which limits their usefulness beyond exploratory purposes.

# Conclusion

This project explored a complete machine learning workflow for
predicting forest cover types using only cartographic features. Through
preprocessing, feature transformation, and systematic evaluation of
several supervised and unsupervised models, we gained a clear
understanding of how different algorithms interpret this
high-dimensional environmental dataset. Among the supervised approaches,
neural networks performed best, followed by logistic regression, while
svm struggled with the strong non-linear patterns present in the data.
Unsupervised models, as expected, could not match supervised accuracy
because they lacked access to class labels, but they were still helpful
for studying natural group structure. The PowerTransformer played a key
role in improving distribution symmetry and model stability, especially
for linear methods. Stratified sampling ensured fair evaluation despite
class imbalance. Overall, the results highlight the importance of
thoughtful preprocessing and the advantages of non-linear models when
dealing with complex ecological relationships. The study demonstrates
how machine learning can effectively support environmental analysis and
provides a foundation for more advanced modeling in the future.

# References

## Web Resources

- <https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals>

- <https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset>

- <https://scikit-learn.org/stable/>

- <https://pandas.pydata.org/docs/>

- <https://numpy.org/doc/stable/>

- <https://matplotlib.org/stable/contents.html>

- <https://seaborn.pydata.org/>

- <https://docs.python.org/3/>
