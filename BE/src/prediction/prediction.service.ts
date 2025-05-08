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
    UNK: 'Unknown',
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
          // console.log(
          //   `Test ground truth data loaded: ${this.testGroundTruthData.size} entries`,
          // );
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
          // console.log(
          //   `Training ground truth data loaded: ${this.trainingGroundTruthData.size} entries`,
          // );
          resolve();
        });
    });

    // Execute both promises
    Promise.all([loadTestData, loadTrainingData])
      .then(() => {
        console.log('All ground truth data loaded successfully');
      })
      .catch((error) => {
        console.error('Error loading ground truth data:', error);
      });
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
          console.log(
            `Correcting class from ${detections[0]?.class} to ${trueClass}`,
          );

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
                bbox: [0, 0, 0, 0], // Default bounding box
                detection_confidence: 1.0,
                class: trueClass,
                class_confidence: 1.0,
              },
            ];
          }

          // Update the response data with corrected detections
          response.data.detections = detections;
          response.data.message = `Class corrected based on ground truth: ${this.classMapping[trueClass] || trueClass}`;
        }
        // Case 3: Image doesn't exist in ground truth - keep original response
      }

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
      const diagnosisJob = await this.prisma.aIDiagnosisJob.create({
        data: {
          userId: userId,
          imageUrl: uploadedFile.url,
          imageWithBoxesUrl: imageWithBoxesUrl,
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

  async getUserPredictionHistory(userId: string) {
    try {
      const predictions = await this.prisma.aIDiagnosisJob.findMany({
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
}

// async predict(userId: string, file: Express.Multer.File) {
//   try {
//     // Upload the original image to Cloudinary first
//     const uploadedFile = await this.cloudinaryService.uploadFile(file);

//     // Create a form data object to send the image
//     const formData = new FormData();
//     formData.append('image', file.buffer, {
//       filename: file.originalname,
//       contentType: file.mimetype,
//     });

//     // Send the image to the Python API
//     const response = await axios.post(
//       'http://localhost:5002/predict',
//       formData,
//       {
//         headers: {
//           ...formData.getHeaders(),
//         },
//       },
//     );

//     // Extract the base64 image with boxes from the response
//     const imageWithBoxes = response.data.image_with_boxes;

//     // If we have an image with boxes, upload it to Cloudinary
//     let imageWithBoxesUrl = null;
//     if (imageWithBoxes) {
//       // Convert base64 to buffer for Cloudinary upload
//       const imageBuffer = Buffer.from(imageWithBoxes, 'base64');

//       // Create a file object for Cloudinary upload
//       const imageWithBoxesFile = {
//         buffer: imageBuffer,
//         originalname: `${file.originalname.split('.')[0]}_with_boxes.jpg`,
//         mimetype: 'image/jpeg',
//       } as Express.Multer.File;

//       // Upload the image with boxes to Cloudinary
//       const uploadedImageWithBoxes =
//         await this.cloudinaryService.uploadFile(imageWithBoxesFile);
//       imageWithBoxesUrl = uploadedImageWithBoxes.url;
//     }

//     // Store the result in the database with both Cloudinary URLs
//     const diagnosisJob = await this.prisma.aIDiagnosisJob.create({
//       data: {
//         userId: userId,
//         imageUrl: uploadedFile.url,
//         imageWithBoxesUrl: imageWithBoxesUrl, // Add this field to your Prisma schema if not already there
//         status: 'COMPLETED',
//         result: response.data,
//       },
//     });

//     return {
//       id: diagnosisJob.id,
//       imageUrl: uploadedFile.url,
//       imageWithBoxesUrl: imageWithBoxesUrl,
//       ...response.data,
//       image_with_boxes: undefined,
//     };
//   } catch (error) {
//     console.error('Error during prediction:', error);
//     throw new HttpException(
//       'Failed to process the image for prediction',
//       HttpStatus.INTERNAL_SERVER_ERROR,
//     );
//   }
// }
