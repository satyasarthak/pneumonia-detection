# Requirements Document

## Introduction

This feature delivers an end-to-end pneumonia detection system built on chest X-ray imaging data from the RSNA Pneumonia Detection dataset. The system performs three-class classification of chest radiographs, distinguishing between Normal, Lung Opacity (Pneumonia), and No Lung Opacity / Not Normal cases. The scope covers the full machine learning lifecycle: data ingestion and exploration, DICOM preprocessing, a baseline convolutional neural network trained from scratch, transfer learning with at least two pretrained architectures, model comparison and selection, and full deployment as a Streamlit application containerized with Docker and served through GitHub Codespaces with live inference. The system also produces actionable insights and a business-quality report to support clinical decision-support recommendations.

## Glossary

- **System**: The complete pneumonia detection software solution, including data pipeline, models, and deployment application.
- **Data_Pipeline**: The component responsible for importing, inspecting, preprocessing, and splitting the imaging dataset.
- **Label_Source**: The `stage_2_detailed_class_info.csv` file, which provides the three-class label for each patient identifier.
- **Class_Label**: One of exactly three categories — Normal, Lung Opacity (Pneumonia), or No Lung Opacity / Not Normal.
- **Patient_Id**: The unique `patientId` field associated with each chest X-ray study in the dataset.
- **DICOM_Image**: A chest radiograph stored in the DICOM medical imaging format within the training and test image archives.
- **EDA_Module**: The component that performs exploratory data analysis, including visualization and class distribution inspection.
- **Baseline_CNN**: A convolutional neural network model built and trained from scratch as a performance baseline.
- **Transfer_Model**: A classification model built using a pretrained CNN backbone (VGG16, ResNet50, or MobileNetV2) with a custom classification head.
- **Model_Evaluator**: The component that computes and reports multi-class evaluation metrics for each trained model.
- **Model_Registry**: The serialized-model artifact store used to save and reload trained models for inference.
- **Streamlit_App**: The web application that accepts an uploaded chest X-ray image and returns a predicted class with probabilities.
- **Deployment_Package**: The Docker-based packaging and configuration used to run the Streamlit_App in a container.
- **Codespaces_Deployment**: The running instance of the Deployment_Package on GitHub Codespaces exposed through a forwarded URL.
- **Report_Module**: The deliverable that documents insights, recommendations, and business context.

## Requirements

### Requirement 1: Data Overview and Ingestion

**User Story:** As a data scientist, I want to import the imaging dataset and inspect its structure, so that I can confirm the data is loaded correctly before analysis.

#### Acceptance Criteria

1. WHEN the Data_Pipeline loads the Label_Source, THE Data_Pipeline SHALL parse each record into a Patient_Id and its associated Class_Label.
2. WHEN the Data_Pipeline loads the label and image data, THE Data_Pipeline SHALL report the number of records and the dimensions of each loaded table.
3. WHERE a Patient_Id appears in multiple records of the Label_Source, THE Data_Pipeline SHALL de-duplicate the Patient_Id to a single Class_Label per Patient_Id.
4. THE Data_Pipeline SHALL restrict the set of Class_Label values to exactly three categories: Normal, Lung Opacity (Pneumonia), and No Lung Opacity / Not Normal.
5. IF a referenced DICOM_Image file cannot be located in the image archives, THEN THE Data_Pipeline SHALL record the missing Patient_Id and exclude that record from downstream processing.

### Requirement 2: Exploratory Data Analysis

**User Story:** As a data scientist, I want to explore sample images and the class distribution, so that I can understand the dataset characteristics and identify imbalance.

#### Acceptance Criteria

1. WHEN the EDA_Module runs, THE EDA_Module SHALL display a set of randomly selected chest X-ray images for each of the three Class_Label categories.
2. WHEN the EDA_Module displays a sample image, THE EDA_Module SHALL annotate the image with its Class_Label.
3. WHEN the EDA_Module analyzes the dataset, THE EDA_Module SHALL compute and display the count of Patient_Id records per Class_Label.
4. WHEN the EDA_Module reports the class distribution, THE EDA_Module SHALL record observations that identify whether class imbalance is present.

### Requirement 3: Data Preprocessing

**User Story:** As a data scientist, I want DICOM images decoded, normalized, and split into datasets, so that they are ready for model training.

#### Acceptance Criteria

1. WHEN the Data_Pipeline processes a DICOM_Image, THE Data_Pipeline SHALL decode the DICOM_Image into a pixel array.
2. WHEN the Data_Pipeline converts image color representation, THE Data_Pipeline SHALL transform the image between RGB and grayscale as required by the target model input format.
3. WHEN the Data_Pipeline preprocesses a sample image, THE Data_Pipeline SHALL display the image before and after preprocessing for visual comparison.
4. WHEN the Data_Pipeline prepares the dataset, THE Data_Pipeline SHALL partition the de-duplicated records into training, validation, and test subsets.
5. WHEN the Data_Pipeline partitions the dataset, THE Data_Pipeline SHALL preserve the proportion of each Class_Label across the training, validation, and test subsets.
6. WHEN the Data_Pipeline prepares image pixel data for a model, THE Data_Pipeline SHALL normalize pixel values to a defined numeric range.
7. WHEN the Data_Pipeline feeds images to a model, THE Data_Pipeline SHALL supply images through a batched generator that loads images incrementally rather than loading the full dataset into memory at once.

### Requirement 4: Baseline Model Building

**User Story:** As a data scientist, I want to build and train a CNN from scratch, so that I have a baseline to compare transfer learning against.

#### Acceptance Criteria

1. THE Baseline_CNN SHALL produce a three-node output corresponding to the three Class_Label categories.
2. WHEN the Baseline_CNN is trained, THE Data_Pipeline SHALL supply the training and validation subsets to the training process.
3. WHEN Baseline_CNN training completes, THE Model_Evaluator SHALL compute multi-class evaluation metrics on the test subset.
4. WHEN the Model_Evaluator reports Baseline_CNN results, THE Model_Evaluator SHALL record commentary interpreting the Baseline_CNN performance.

### Requirement 5: Transfer Learning and Model Selection

**User Story:** As a data scientist, I want to train multiple pretrained architectures and select the best, so that I maximize classification performance.

#### Acceptance Criteria

1. THE System SHALL build at least two Transfer_Model instances, each using a distinct pretrained backbone selected from VGG16, ResNet50, and MobileNetV2.
2. WHEN a Transfer_Model is constructed, THE System SHALL attach a custom classification head that produces a three-node output corresponding to the three Class_Label categories.
3. WHEN each Transfer_Model is trained, THE Model_Evaluator SHALL compute multi-class evaluation metrics on the test subset.
4. WHEN all models have been evaluated, THE Model_Evaluator SHALL present a comparison of the Baseline_CNN and every Transfer_Model across the same evaluation metrics.
5. WHEN the comparison is complete, THE System SHALL select one model as the best performer and record the rationale for the selection.
6. WHEN the best model is selected, THE Model_Registry SHALL serialize the selected model to a persistent artifact.
7. WHEN a serialized model is reloaded from the Model_Registry, THE System SHALL perform inference on a test image and produce a predicted Class_Label.

### Requirement 6: Model Deployment

**User Story:** As an end user, I want to upload a chest X-ray through a web app and receive a prediction, so that I can obtain a pneumonia assessment without running code.

#### Acceptance Criteria

1. WHEN a user uploads a chest X-ray image to the Streamlit_App, THE Streamlit_App SHALL return the predicted Class_Label for the uploaded image.
2. WHEN the Streamlit_App returns a prediction, THE Streamlit_App SHALL display the probability associated with each Class_Label.
3. IF an uploaded file is not a supported chest X-ray image format, THEN THE Streamlit_App SHALL return an error message and reject the upload.
4. THE Deployment_Package SHALL declare the backend and frontend dependencies required to run the Streamlit_App.
5. THE Deployment_Package SHALL provide a Docker configuration that builds and runs the Streamlit_App in a container.
6. WHEN the Deployment_Package is published, THE System SHALL push the application code and configuration to a source code repository.
7. WHEN the Codespaces_Deployment is started, THE Codespaces_Deployment SHALL expose the Streamlit_App through a forwarded URL.
8. WHEN a user accesses the Codespaces_Deployment through the forwarded URL, THE Codespaces_Deployment SHALL perform live inference on an uploaded image and return a predicted Class_Label with probabilities.

### Requirement 7: Actionable Insights and Recommendations

**User Story:** As a business stakeholder, I want documented insights and recommendations, so that I can understand how the model supports clinical decisions.

#### Acceptance Criteria

1. WHEN model evaluation is complete, THE Report_Module SHALL document actionable insights derived from the model results and the data analysis.
2. WHEN the Report_Module documents insights, THE Report_Module SHALL provide recommendations for applying the pneumonia detection results in a clinical decision-support context.

### Requirement 8: Business Report Quality

**User Story:** As a business stakeholder, I want a professionally structured report, so that the findings are clear and usable by a non-technical audience.

#### Acceptance Criteria

1. THE Report_Module SHALL present the problem definition, methodology, results, and recommendations in a structured document.
2. THE Report_Module SHALL express results and observations in language accessible to a non-technical business audience.
3. WHEN the Report_Module presents quantitative results, THE Report_Module SHALL include supporting visualizations of the data analysis and model performance.
