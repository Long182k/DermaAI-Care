# NGUYENTHANHLONG-24072179-UWE Thesis 
# AI-POWERED DERMATOLOGICAL APPOINTMENT BOOKING AND SKIN CANCER PREDICTION WEBSITE

## Resources

- **Training Dataset:** [Mapping-ISIC2019-Training-Dataset](https://www.kaggle.com/datasets/longngg/mapping-isic2019-training-dataset)
- **Testing Dataset:** [Mapping-ISIC2019-Testing-Dataset](https://www.kaggle.com/datasets/longngg/mapping-isic2019-testing-dataset)
- **Skin Lesion Classification Model:** [Long Ngg | Skin-Lesion-Classification-Model | Kaggle](https://www.kaggle.com/models/longngg/skin-lesion-classification-model)
- **Skin Lesion Detection Model (YOLOv11):** [Long Nguyenzzz | yolov11-skinz | Kaggle](https://www.kaggle.com/models/longnguyenzzz/yolov11-skinz/pyTorch/default/1)

DermaAI-Care is a comprehensive platform designed to assist in the early detection of skin cancer and streamline dermatological consultations. It leverages advanced machine learning models for image analysis and provides features for online appointment booking and real-time communication between patients and doctors.

## Project Aim

The primary goal of DermaAI-Care is to alleviate the burden on healthcare systems by providing an accessible and efficient tool for preliminary skin cancer screening. By enabling users to get quick AI-driven analysis of skin lesions and connect with dermatologists remotely, we aim to facilitate early diagnosis and timely medical intervention.

## Key Features

- **AI-Powered Skin Lesion Analysis:**
  - **Detection Model:** Utilizes a YOLOv11-based model for detecting suspicious regions in skin images.
  - **Classification Model:** Employs an SE-ResNeXt101(32x4d) model to classify detected lesions into various skin condition categories, including melanoma, nevus, basal cell carcinoma, etc.
- **Online Appointment Booking:** Allows patients to schedule consultations with dermatologists seamlessly.
- **Real-Time Communication:** Facilitates direct messaging and potential video consultations between patients and doctors.
- **Prediction History:** Users can track their past analyses and share them with their healthcare providers.

## Technical Stack

- **Frontend (FE):** React, TypeScript, TailwindCSS
- **Backend (BE):** NestJS, TypeScript, PostgreSQL
- \*\*Machine Learning (Training):
  - Detection: YOLOv11 (based on YOLOv11 architecture)
  - Classification: SE-ResNeXt101(32x4d)
  - Frameworks: PyTorch

## Performance Metrics

- **Classification Model (AUC):** 92.35%
- **Detection Model (mAP):** ~88%

## Modules

### Backend (BE)

Built with NestJS, the backend handles:

- User Authentication & Authorization (JWT-based)
- Database Management (PostgreSQL with Prisma ORM)
- API endpoints for predictions, appointments, user management, and statistics.
- Real-time communication (potentially using WebSockets).
- Image processing and interaction with the ML models.

### Frontend (FE)

Developed with React, the frontend provides the user interface for:

- Image upload and AI analysis requests.
- Displaying prediction results and explanations.
- User registration and profile management.
- Appointment booking and management.
- Communication interface (chat/messaging).

### Training (Machine Learning Models)

Contains the scripts and notebooks for training and evaluating the skin lesion detection and classification models.

- **YOLOv11:** Leverages the Ultralytics YOLOv11 framework (based on YOLOv5 architecture) for object detection model training. The models are trained on custom datasets of skin lesion images.
- **SE-ResNeXt101(32x4d):** Custom classification model training scripts.

## Getting Started

### Install dependencies and run sources

a. Install dependencies command: `pnpm install` (you can also use `npm` or `yarn`)

b. Run source:

i. **Frontend:** 
```bash
pnpm dev
```
ii. **Backend:** 

```bash
npx prisma generate 
```
(This command will read your `schema.prisma` file and generate the Prisma Client, providing type-safe database access and queries) 

```bash
pnpm start:dev
```

**Note:** Run the backend before starting the frontend.

### Setting up Stripe

1.  Log in to your Stripe account in the terminal:
    ```bash
    stripe login
    ```
2.  Listen for webhook events and forward them to your local backend. Take note of the `webhook_key` provided in the output and add it to your `.env` file in the `BE` directory.
    ```bash
    stripe listen --forward-to localhost:3000/payment/webhook
    ```

### Backend Setup Example (from BE/README.md)

```bash
# Clone the repository (if not already done)
# git clone https://github.com/Long182k/DermaAI-Care.git
cd BE

# Install dependencies (example using pnpm)
# pnpm install

# Set up your .env file with database credentials, JWT secret, etc.

# Start the development server
# pnpm start:dev
```

## YouTube Video Demo

Watch a demonstration of the project on YouTube: [https://www.youtube.com/watch?v=Jd8M4p1yTCQ](https://www.youtube.com/watch?v=Jd8M4p1yTCQ)
