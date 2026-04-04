# Ignisia Grading Model

Ignisia Grading Model is an AI-assisted answer sheet evaluation platform designed to reduce the manual effort involved in checking large volumes of student responses while still keeping teachers in control of the final assessment process. The system accepts uploaded answer sheets in image, PDF, or ZIP form, performs OCR to extract student responses, and organizes answers into semantic clusters so that similar responses can be reviewed together instead of one by one. These clusters are then compared against an uploaded answer key using rubric-based evaluation, semantic similarity scoring, and concept-level matching to suggest marks, reasons, and confidence levels for each answer group.

The platform is built for practical classroom workflows rather than fully opaque automation. Teachers can inspect cluster-level outputs, review student answer scripts, override suggested marks where needed, and identify low-confidence or unusual responses through dedicated outlier views. Ignisia also supports multilingual evaluation, allowing semantically similar answers written in English and regional languages to be grouped together where appropriate. In addition, the platform can generate personalized tutoring feedback for students, including concept-specific explanations, practice questions, and export-ready output formats.

## Features

- OCR-based extraction from student answer sheets
- Support for PNG, JPG, JPEG, PDF, and ZIP uploads
- Semantic clustering of similar answers
- Multilingual clustering support for English and regional-language answers
- Rubric-based grading with semantic similarity scoring
- Suggested marks, reasons, and confidence levels per cluster
- Manual override workflow for teachers
- Outlier detection for uncertain or unusual clusters
- Cluster drilldown to inspect answer sheets inside each cluster
- Personalized student feedback generation
- Exportable JSON, email-text, and PDF feedback outputs
- Cost and efficiency reporting for pipeline runs
- Streamlit-based review dashboard

## How It Works
### 1. OCR
Uploaded answer sheets are processed using OCR to extract student responses. The OCR pipeline supports flexible file grouping and can work with folder-based datasets, ZIP files, PDFs, or individual image uploads.

### 2. Answer Extraction
The backend reads the uploaded answer key and uses the detected question structure to extract answer text question-wise from each student sheet.

### 3. Semantic Clustering
For each question, answers are embedded using a multilingual sentence-transformer model and grouped into clusters based on semantic similarity. This allows answers written in different wording, and in some cases different languages, to be grouped together if they convey similar meaning.

### 4. Rubric-Based Grading
Each cluster is compared against answer-key variations and rubric requirements. The system generates:

suggested marks
cluster-level reason
confidence score
matched and missing concept indicators

### 5. Teacher Review
Teachers can inspect clusters, view answer-sheet images, review low-confidence clusters, override marks and reasons, and focus on outlier groups that need human attention.

### 6. Student Feedback Generation
After review, Ignisia Grading Model can generate personalized tutoring feedback for students, including:

short explanatory feedback
concept recap
practice question
exportable email text
exportable PDF output
Multilingual Support
Ignisia includes multilingual-aware preprocessing and semantic comparison so that answers written in English and regional languages can still be compared meaningfully. The system uses:

script-aware OCR cleanup
multilingual text normalization
multilingual sentence embeddings
semantic grading rather than simple string matching
This improves clustering and grading quality for mixed-language answer sets.

Streamlit Interface
The Streamlit app provides three main pages:

Home
Used to upload:

answer key
optional manifest
answer sheets as a single file or multiple images
Dashboard
Shows:

cluster overview
confidence and marks information
script distribution
answer-sheet drilldown
cost and efficiency metrics
Override Review
Allows teachers to:

inspect clusters
review answer-sheet images
check similarity and keyword analysis
override marks and reasons
generate tutoring feedback packages
Outliers
Shows clusters that need more attention, such as:

low-confidence clusters
weak semantic matches
noise or uncertain answer groups

## Supported Inputs
Answer sheets
PNG
JPG
JPEG
PDF
ZIP
multiple uploaded images in the Streamlit UI
Answer key
CSV

## Outputs
Depending on the workflow, Ignisia generates:

OCR JSON and CSV
clustered answer CSV and JSON
grading output JSON
reviewed output JSON
cost and efficiency logs
student feedback JSON
student feedback PDFs
student email text exports
Review and Outlier Logic
Ignisia separates clustering confidence from grading confidence.

A cluster may be flagged for review if:

it is a clustering outlier
its semantic similarity to the answer key is weak
its confidence is below the configured threshold
This helps teachers focus attention on uncertain answer groups without having to manually inspect every script.

## Current Strengths
End-to-end OCR to review pipeline
Streamlit-based teacher workflow
Cluster-based grading assistance
multilingual semantic support
exportable tutoring feedback
isolated run-based outputs to avoid mixing old and new uploads

## Current Limitations
OCR quality still depends on image clarity and handwriting quality
question extraction can still be heuristic for highly irregular answer-sheet layouts
clustering on very small datasets may be unstable
answer-key structure still works best when question boundaries are clear
personalized feedback is currently generated by a local rule-based engine unless extended further

## Recommended Usage
Ignisia works best when:

answer sheets are clearly scanned
answer key quality is high
the uploaded dataset matches the uploaded answer key
teachers review low-confidence clusters before finalizing marks

## Future Improvements
Potential future enhancements include:

stronger question segmentation for arbitrary exam formats
better support for more regional languages
richer analytics for teacher review
database-backed run history
stronger automated feedback generation
deployment-ready backend API layer

## Project Structure

```text
Ignisia-main/
├── streamlit_app.py
├── app_utils.py
├── ocr_final.py
├── requirements.txt
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Override_Review.py
│   └── 3_Outliers.py
├── backend/
│   ├── full_pipeline.py
│   ├── embedding.py
│   ├── run_pipeline.py
│   ├── csv_loader.py
│   ├── cluster_processor.py
│   ├── rubric_generator.py
│   ├── text_normalizer.py
│   ├── cost_efficiency_logger.py
│   ├── feedback_generator.py
│   ├── generate_student_feedback.py
│   ├── review_store.py
│   ├── llm_client.py
│   ├── pdf_exporter.py
│   ├── email_exporter.py
│   ├── grading/
│   │   ├── __init__.py
│   │   ├── scoring_engine.py
│   │   ├── keyword_matcher.py
│   │   ├── math_validator.py
│   │   └── regex_parser.py
│   └── README.md
└── uploads/
