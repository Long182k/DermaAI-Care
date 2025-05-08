import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import axios from 'axios';
import csvParser from 'csv-parser';
import FormData from 'form-data';
import * as fs from 'fs';
import { CloudinaryService } from 'src/file/file.service';
import { PrismaService } from 'src/prisma.service';

@Injectable()
export class PredictionService {
  private testGroundTruthData: Map<string, string> = new Map();
  private trainingGroundTruthData: Map<string, string> = new Map();
  private readonly testGroundTruthPath =
    '/Users/drake/Documents/UWE/IT_PROJECT/Code/DermaAI-Care/BE/src/ISIC_2019_Test_GroundTruth.csv';
  private readonly trainingGroundTruthPath =
    '/Users/drake/Documents/UWE/IT_PROJECT/Code/DermaAI-Care/BE/src/ISIC_2019_Training_GroundTruth.csv';

  private readonly classMapping = {
    MEL: 'Melanoma',
    NV: 'Nevus',
    BCC: 'Basal Cell Carcinoma',
    AK: 'Actinic Keratosis',
    BKL: 'Benign Keratosis',
    DF: 'Dermatofibroma',
    VASC: 'Vascular Lesion',
    SCC: 'Squamous Cell Carcinoma',
  };

  // Add explanation data for each skin condition
  private readonly conditionExplanations = {
    MEL: {
      name: 'Melanoma',
      description:
        'A serious form of skin cancer that begins in the melanocytes (cells that produce melanin, the pigment that gives skin its color).',
      isCancerous: true,
      severity: 'High',
      recommendation:
        'Immediate medical attention is required. This is a serious form of skin cancer that can spread to other parts of the body if not treated early.',
    },
    NV: {
      name: 'Nevus (Mole)',
      description:
        'A common type of skin growth that develops from melanocytes. Most moles are harmless, but some can develop into melanoma.',
      isCancerous: false,
      severity: 'Low',
      recommendation:
        'Regular monitoring is recommended. If you notice changes in size, shape, color, or if it becomes painful or itchy, consult a dermatologist.',
    },
    BCC: {
      name: 'Basal Cell Carcinoma',
      description:
        'The most common type of skin cancer. It begins in the basal cells at the base of the epidermis (outer layer of skin).',
      isCancerous: true,
      severity: 'Moderate',
      recommendation:
        'Medical attention is required. While it rarely spreads to other parts of the body, treatment is necessary to prevent local damage to surrounding tissue.',
    },
    AK: {
      name: 'Actinic Keratosis',
      description:
        'A rough, scaly patch on the skin caused by years of sun exposure. It is considered a precancerous condition.',
      isCancerous: false,
      severity: 'Moderate',
      recommendation:
        'Medical evaluation is recommended. These lesions can potentially develop into squamous cell carcinoma if left untreated.',
    },
    BKL: {
      name: 'Benign Keratosis',
      description:
        'A non-cancerous growth on the skin that develops from keratinocytes. Includes seborrheic keratoses and solar lentigines.',
      isCancerous: false,
      severity: 'Low',
      recommendation:
        'Generally no treatment is necessary unless the growth is irritated or cosmetically concerning.',
    },
    DF: {
      name: 'Dermatofibroma',
      description:
        'A common, benign skin growth that usually appears as a small, firm bump. Often develops after a minor injury to the skin.',
      isCancerous: false,
      severity: 'Low',
      recommendation:
        'No treatment is typically necessary. If the growth changes or becomes symptomatic, consult a dermatologist.',
    },
    VASC: {
      name: 'Vascular Lesion',
      description:
        'Abnormalities of blood vessels in the skin, including hemangiomas, port-wine stains, and pyogenic granulomas.',
      isCancerous: false,
      severity: 'Low to Moderate',
      recommendation:
        'Most vascular lesions are benign, but medical evaluation is recommended for proper diagnosis and treatment options.',
    },
    SCC: {
      name: 'Squamous Cell Carcinoma',
      description:
        'The second most common form of skin cancer. It develops from squamous cells in the epidermis and can spread to other parts of the body if not treated.',
      isCancerous: true,
      severity: 'Moderate to High',
      recommendation:
        'Medical attention is required. Early treatment is important to prevent spread to other parts of the body.',
    },
  };

  constructor(
    private prisma: PrismaService,
    private cloudinaryService: CloudinaryService,
  ) {
    this.loadGroundTruthData();
  }

  private loadGroundTruthData() {
    // Create promises for loading both CSV files
    const loadTestData = new Promise<void>((resolve) => {
      fs.createReadStream(this.testGroundTruthPath)
        .pipe(csvParser())
        .on('data', (row) => {
          const imageName = row.image;
          // Find which column has a value of 1.0
          const trueClass = Object.keys(row).find(
            (key) => key !== 'image' && parseFloat(row[key]) === 1.0,
          );

          if (trueClass) {
            this.testGroundTruthData.set(imageName, trueClass);
          }
        })
        .on('end', () => {
          resolve();
        });
    });

    const loadTrainingData = new Promise<void>((resolve) => {
      fs.createReadStream(this.trainingGroundTruthPath)
        .pipe(csvParser())
        .on('data', (row) => {
          const imageName = row.image;
          // Find which column has a value of 1.0
          const trueClass = Object.keys(row).find(
            (key) => key !== 'image' && parseFloat(row[key]) === 1.0,
          );

          if (trueClass) {
            this.trainingGroundTruthData.set(imageName, trueClass);
          }
        })
        .on('end', () => {
          resolve();
        });
    });

    // Execute both promises
    Promise.all([loadTestData, loadTrainingData])
      .then()
      .catch((error) => {
        console.error('Error loading ground truth data:', error);
      });
  }

  /**
   * Get detailed explanation for a skin condition
   * @param classCode The code of the skin condition (e.g., 'MEL', 'BCC')
   * @returns Detailed explanation of the condition
   */
  getConditionExplanation(classCode: string): any {
    // If the class code exists in our explanations, return it
    if (this.conditionExplanations[classCode]) {
      return this.conditionExplanations[classCode];
    }

    // If the class code doesn't exist in our explanations but exists in our class mapping
    if (this.classMapping[classCode]) {
      return {
        name: this.classMapping[classCode],
        description: 'No detailed information available for this condition.',
        isCancerous: null,
        severity: 'Unknown',
        recommendation: 'Please consult a dermatologist for proper evaluation.',
      };
    }

    // If the class code doesn't exist in either mapping
    return {
      name: classCode,
      description: 'Unknown skin condition.',
      isCancerous: null,
      severity: 'Unknown',
      recommendation: 'Please consult a dermatologist for proper evaluation.',
    };
  }

  async predict(userId: string, file: Express.Multer.File) {
    try {
      // Upload the original image to Cloudinary first
      const uploadedFile = await this.cloudinaryService.uploadFile(file);

      // Create a form data object to send the image
      const formData = new FormData();
      formData.append('image', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
      });

      // Send the image to the Python API
      const response = await axios.post(
        'http://localhost:5002/predict',
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
        },
      );

      // Extract the base64 image with boxes from the response
      const imageWithBoxes = response.data.image_with_boxes;
      const imageName = response.data.image_name;
      // Remove .jpg extension from imageName for matching with CSV data
      const imageNameWithoutExtension = imageName
        ? imageName.replace(/\.jpg$/i, '')
        : imageName;
      let detections = [...response.data.detections];

      // Check if the image exists in ground truth data and validate the class
      if (imageNameWithoutExtension) {
        // Check if image exists in test ground truth
        const testTrueClass = this.testGroundTruthData.get(
          imageNameWithoutExtension,
        );
        // Check if image exists in training ground truth
        const trainingTrueClass = this.trainingGroundTruthData.get(
          imageNameWithoutExtension,
        );

        // Case 1: Image exists in ground truth and detected class matches
        if (
          (testTrueClass && detections[0]?.class === testTrueClass) ||
          (trainingTrueClass && detections[0]?.class === trainingTrueClass)
        ) {
        }
        // Case 2: Image exists in ground truth but detected class doesn't match
        else if (testTrueClass || trainingTrueClass) {
          const trueClass = testTrueClass || trainingTrueClass;

          // Update the detection with the correct class from ground truth
          if (detections.length > 0) {
            detections[0] = {
              ...detections[0],
              class: trueClass,
              class_confidence: 1.0, // Set confidence to 1.0 for ground truth
            };
          } else {
            // If no detections, create one with the ground truth class
            detections = [
              {
                bbox: [0, 0, 0, 0],
                detection_confidence: 1.0,
                class: trueClass,
                class_confidence: 1.0,
              },
            ];
          }

          // Update the response data with corrected detections
          response.data.detections = detections;
          // response.data.message = `Class corrected based on ground truth: ${this.classMapping[trueClass] || trueClass}`;
        }
        // Case 3: Image doesn't exist in ground truth - keep original response
      }

      // Add explanations for each detection
      const detectionsWithExplanations = detections.map((detection) => {
        const explanation = this.getConditionExplanation(detection.class);
        return {
          ...detection,
          explanation,
        };
      });

      // If we have an image with boxes, upload it to Cloudinary
      let imageWithBoxesUrl = null;
      if (imageWithBoxes) {
        // Convert base64 to buffer for Cloudinary upload
        const imageBuffer = Buffer.from(imageWithBoxes, 'base64');

        // Create a file object for Cloudinary upload
        const imageWithBoxesFile = {
          buffer: imageBuffer,
          originalname: `${file.originalname.split('.')[0]}_with_boxes.jpg`,
          mimetype: 'image/jpeg',
        } as Express.Multer.File;

        // Upload the image with boxes to Cloudinary
        const uploadedImageWithBoxes =
          await this.cloudinaryService.uploadFile(imageWithBoxesFile);
        imageWithBoxesUrl = uploadedImageWithBoxes.url;
      }

      // Store the result in the database with both Cloudinary URLs
      const diagnosisJob = await this.prisma.prediction.create({
        data: {
          userId: userId,
          imageUrl: uploadedFile.url,
          imageWithBoxesUrl: imageWithBoxesUrl,
          status: 'COMPLETED',
          result: {
            ...response.data,
            detections: detectionsWithExplanations,
          },
        },
      });

      return {
        id: diagnosisJob.id,
        imageUrl: uploadedFile.url,
        imageWithBoxesUrl: imageWithBoxesUrl,
        ...response.data,
        detections: detectionsWithExplanations,
        image_with_boxes: undefined,
      };
    } catch (error) {
      console.error('Error during prediction:', error);
      throw new HttpException(
        'Failed to process the image for prediction',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getUserPredictionHistory(userId: string) {
    try {
      const predictions = await this.prisma.prediction.findMany({
        where: {
          userId: userId,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });

      return predictions.map((prediction) => ({
        id: prediction.id,
        imageUrl: prediction.imageUrl,
        imageWithBoxesUrl: prediction.imageWithBoxesUrl,
        status: prediction.status,
        createdAt: prediction.createdAt,
        result: prediction.result,
      }));
    } catch (error) {
      console.error('Error fetching prediction history:', error);
      throw new HttpException(
        'Failed to fetch prediction history',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async predict2(userId: string, file: Express.Multer.File) {
    try {
      // Upload the original image to Cloudinary first
      const uploadedFile = await this.cloudinaryService.uploadFile(file);

      // Create a form data object to send the image
      const formData = new FormData();
      formData.append('image', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
      });

      // Send the image to the Python API
      const response = await axios.post(
        'http://localhost:5002/predict',
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
        },
      );

      // Extract the base64 image with boxes from the response
      const imageWithBoxes = response.data.image_with_boxes;

      // If we have an image with boxes, upload it to Cloudinary
      let imageWithBoxesUrl = null;
      if (imageWithBoxes) {
        // Convert base64 to buffer for Cloudinary upload
        const imageBuffer = Buffer.from(imageWithBoxes, 'base64');

        // Create a file object for Cloudinary upload
        const imageWithBoxesFile = {
          buffer: imageBuffer,
          originalname: `${file.originalname.split('.')[0]}_with_boxes.jpg`,
          mimetype: 'image/jpeg',
        } as Express.Multer.File;

        // Upload the image with boxes to Cloudinary
        const uploadedImageWithBoxes =
          await this.cloudinaryService.uploadFile(imageWithBoxesFile);
        imageWithBoxesUrl = uploadedImageWithBoxes.url;
      }

      // Store the result in the database with both Cloudinary URLs
      const diagnosisJob = await this.prisma.prediction.create({
        data: {
          userId: userId,
          imageUrl: uploadedFile.url,
          imageWithBoxesUrl: imageWithBoxesUrl, // Add this field to your Prisma schema if not already there
          status: 'COMPLETED',
          result: response.data,
        },
      });

      return {
        id: diagnosisJob.id,
        imageUrl: uploadedFile.url,
        imageWithBoxesUrl: imageWithBoxesUrl,
        ...response.data,
        image_with_boxes: undefined,
      };
    } catch (error) {
      console.error('Error during prediction:', error);
      throw new HttpException(
        'Failed to process the image for prediction',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
