# Multimodal Crisis Classification

This project addresses the challenge of classifying crisis-related content from social media using both textual and visual modalities. It utilizes the CrisisMMD dataset, which contains multimodal (text and image) data from various natural disasters, including hurricanes, earthquakes, and floods. The goal of this project is to develop and evaluate a multimodal classification model capable of performing tasks related to humanitarian aid, such as identifying informative content, categorizing humanitarian issues, and assessing damage severity.

# Table of contents
- [Dataset Overview](#dataset-overview)

- [Explorative Dataset Analysis](#explorative-dataset-analysis)

- [Data Preprocessing and Preparation](#data-processing-and-preparation)

- [Unimodal Classification](#unimodal-classification)

- [Multimodal Classification](#multimodal-classification)

- [How to Use](#how-to-use)

- [Applications](#applications)

## Dataset Overview
The **CrisisMMD dataset** is sourced from Twitter during seven major natural disasters in 2017:

* **Hurricane Irma**
* **Hurricane Harvey**
* **Hurricane Maria**
* **Mexico Earthquake**
* **California Wildfires**
* **Iraq-Iran Earthquake**
* **Sri Lanka Floods**

The dataset includes both textual and image data, with annotations that help classify information in multiple humanitarian categories. The key annotations include:

* **Task 1**: Informative vs. Not Informative (whether the content is useful for humanitarian aid)
* **Task 2**: Humanitarian Categories (e.g., infrastructure damage, rescue efforts, affected individuals)
* **Task 3**: Damage Severity Assessment (e.g., severe, mild, or no damage)

The dataset is publicly available and can be accessed [here](https://crisisnlp.qcri.org/crisismmd).

### Citation
- Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. Proceedings of the 12th International Conference on Web and Social Media (ICWSM 2018), AAAI Press.

- Ofli, F., Alam, F., & Imran, M. (2020). Multimodal Deep Learning for Disaster Response: Analysis of Social Media Data. Proceedings of the 17th ISCRAM Conference.

## Explorative Dataset Analysis
We run an in-depth analysis on the dataset which you can replicate with the notebook in the [**eda folder**](./eda/).

### Important Findings
Through our analysis of the dataset, we identified that there are **duplicated images with different IDs in the CrisisMMD dataset**.
<figure align=center>
    <img src=images\duplicate_images_distribution.png width="75%">
    <figcaption>Number of duplicated images per hash value.</figcaption>
</figure>

<figure align=center>
    <img src=images/duplicated_images.png width="75%">
    <figcaption>Example of a same image having different IDs.</figcaption>
</figure>

This issue introduces biases in model training and evaluation, as the same image may appear in both the training and test sets, which can artificially inflate the performance.

This discovery was **not previously known by the authors of the dataset** and is crucial for future work.

## Data Processing and Preparation

This project involves a comprehensive pipeline for processing and preparing the CrisisMMD multimodal dataset, ensuring data quality, consistency, and suitability for various machine learning tasks. The process is divided into two main stages: preprocessing and data preparation.

### Preprocessing ([`01_preprocessing.ipynb`](./01_preprocessing.ipynb))

This stage focuses on cleaning and reconciling the raw CrisisMMD annotations to create a unified and usable dataset. Key steps include:

* **Handling Noisy and Duplicate Entries**: A custom confidence-aware strategy is employed to resolve duplicate tweet entries and images. This strategy balances the frequency and mean confidence of labels to arrive at the most accurate representation.
* **Merging Low-Frequency Categories**: To address class imbalance and sparsity, rare humanitarian categories are aggregated. For example, `injured_or_dead_people` and `missing_or_found_people` are merged into `affected_individuals`, and `vehicle_damage` is combined into `infrastructure_and_utility_damage`.
* **Tweet Text Preprocessing**: Tweet text undergoes a series of cleaning operations, including normalization, removal of URLs, mentions, and hashtags, and proper handling of emojis, to prepare it for text-based analysis.

### Data Preparation ([`02_data_preparation.ipynb`](./02_data_preparation.ipynb))

This stage transforms the preprocessed data into formats suitable for training and evaluating machine learning models. The primary goals are to create robust data splits and generate necessary features.

* **Creation of Robust Data Splits**: Reproducible data splits (training, validation, and testing sets) are generated for unimodal text, unimodal image, and multimodal (text + image) classification tasks. This ensures fair and consistent model evaluation.
* **Text Embeddings Generation**: Text embeddings are generated using pre-trained language models such as BERT, RoBERTa, and BERTweet. These embeddings capture the semantic meaning of the tweet text and are crucial inputs for neural network models.
* **Adaptation of Official CrisisMMD Splits**: The notebook adapts the official CrisisMMD data splits to align with the project's requirements, facilitating benchmarking against existing research.
* **Ensuring Data Integrity**: Throughout the data preparation process, careful measures are taken to maintain class balance across all splits and prevent data leakage, thereby ensuring the validity and reliability of experimental results.

Both preprocessed annotations and data splits can be found in the [data folder](./data/).

## Unimodal Classification

Before going all out on multimodal classification, we focused on finding the best (in regards to our computational power capabilities) model for each input modality.

This section defines the comprehensive pipeline used for unimodal classification tasks, encompassing both image and text data. For each modality, a robust methodology was established to prepare the data, leverage powerful pre-trained models as feature extractors (backbones), train custom classifiers, and meticulously evaluate performance.

The general pipeline involves using pre-trained convolutional neural networks (CNNs) for images and transformer models for text to extract rich features. These features then serve as input to a custom Multi-Layer Perceptron (MLP) classifier, which is trained for our specific classification purposes. Following this initial training, the entire model—including the backbone—is fine-tuned by unfreezing relevant layers of the backbone, allowing the pre-trained features to adapt more precisely to our dataset.

Model performance is rigorously evaluated through a suite of metrics and visualizations, including:
* **Classification Reports**: Providing detailed precision, recall, and F1-score for each class.
* **Confusion Matrices**: Visualizing the counts of true and false positives/negatives.
* **ROC-AUC Curves**: Assessing the model's ability to discriminate between classes across various thresholds.
* **Reliability Diagrams**: Evaluating the calibration of the model's predicted probabilities.

### Compared models:
- ### Unimodal Classification - Images ([`03_image_classification.ipynb`](./03_image_classification.ipynb))
    - VGG16
    - ResNet50
    - EfficientNetB0

- ### Unimodal Classification - Text ([`04_text_classification.ipynb`]())
    - BERT
    - RoBERTa
    - BERTweet

## Multimodal Classification

This final section details the multimodal classification approaches, leveraging the strengths of the best-performing unimodal models identified in previous stages: **RoBERTa** for text features and **EfficientNetB0** for image features. These models serve as powerful backbones, whose outputs are then combined using different fusion techniques to enhance classification performance for crisis-related data.

The core idea is to integrate information from both modalities at various stages of the model, aiming to capture complementary signals that improve overall prediction accuracy and robustness. The evaluation of these multimodal models follows the same rigorous approach as the unimodal ones, including detailed classification reports, confusion matrices, ROC-AUC curves, and reliability diagrams to assess performance comprehensively.

We explored three distinct fusion techniques:

- ### Early Fusion Technique ([`05a_multimodal_early_fusion.ipynb`](./05a_multimodal_early_fusion.ipynb))

    <figure align=center>
        <img src=images\Multimodal_early_fusion_architecture.png>
        <figcaption>Early Fusion Multimodal Model architecture overview.</figcaption>
    </figure>

    In this technique, text and image features are concatenated or merged at an early stage of the model architecture, typically after initial feature extraction but before the final classification layers. This allows the model to learn combined representations from both modalities simultaneously from the beginning of the learning process. The fused features are then fed into subsequent layers for multitask learning on both humanitarian and informativeness tasks.

- ### Late Fusion ([`05b_multimodal_late_fusion.ipynb`](./05b_multimodal_late_fusion.ipynb))

    <figure align=center>
        <img src=images\Multimodal_late_fusion_architecture.png width="75%">
        <figcaption>Late Fusion Multimodal Model architecture overview.</figcaption>
    </figure>

    In contrast to early fusion, late fusion combines the predictions or decision scores from independently trained unimodal models. Each modality (text and image) is processed by its own specialized model (e.g., RoBERTa for text, EfficientNetB0 for images), which then generates separate predictions. These individual predictions are subsequently merged, often through weighted averaging or another aggregation strategy, to produce the final multimodal classification. This approach can be beneficial for its modularity and interpretability.

- ### Joint Fusion ([`05c_multimodal_joint_fusion.ipynb`](./05c_multimodal_joint_fusion.ipynb))
    
    <figure align=center>
        <img src=images\Multimodal_joint_fusion_architecture.png>
        <figcaption>Joint Fusion Multimodal Model architecture overview.</figcaption>
    </figure>
    
    While "joint fusion" can sometimes be used interchangeably with early fusion, in this context, it refers to a method where text and image features are integrated into a single, cohesive model that processes both modalities together throughout various layers, not just at an initial concatenation step. This technique aims to capture complex cross-modal interactions by designing a unified network architecture that simultaneously learns and combines representations from both inputs, leading to a more deeply integrated understanding for multitask learning on humanitarian and informativeness tasks.

## How to Use
    
1. **First clone the repository locally:**

    ```sh
    git clone https://github.com/FLaTNNBio/GA-DPAMSA
    ```

2. **Place in the project directory and create a virtual environment (highly recommended)**
    
    ```sh
    python -m venv venv
    ```

3. **Install all required libraries with the following command**
    
    ```
    pip install -r requirements.txt
    ```

4. **Download the dataset: https://crisisnlp.qcri.org/crisismmd (CrisisMMD dataset version v2.0)**

5. **Create a `.env` file with the following path variables:**
    ```bash
    DATASET_DIR = path\to\CrisisMMD_v2.0\
    ANNOTATIONS_DIR = path\to\annotations\
    DATA_IMAGE_DIR = path\to\data_image\
    CRISISMMD_DATASPLITS_DIR = path\to\crisismmd_datasplit\

    PROJECT_DIR = path\to\multimodal-crisis-classification\ 
    DATA_DIR = path\to\data\
    IMAGES_DIR = path\to\images\
    EDA_DIR = path\to\eda\
    EMBEDDINGS_DIR = path\to\embeddings\
    DATASPLITS_DIR = path\to\data\datasplits\
    ADAPTED_DATASPLITS_DIR = path\to\data\crisismmd_adapted_datasplits\
    MODELS_DIR = path\toc\models\
    ```
6. Run [`00_dataset_setup.ipynb`](./00_dataset_setup.ipynb) to load annotations files and merge them into a single one.

You are now ready to go.

## Applications

This project can be used for:

* **Humanitarian Aid**: Classifying and identifying useful content for disaster response.
* **Situational Awareness**: Integrating text and image data to improve decision-making during crises.
* **Multimodal Learning**: Leveraging both text and image features to enhance model accuracy and robustness.
