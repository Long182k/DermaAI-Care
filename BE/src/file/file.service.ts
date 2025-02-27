import { BadRequestException, Injectable } from '@nestjs/common';
import { v2 as cloudinary } from 'cloudinary';

@Injectable()
export class CloudinaryService {
  constructor() {
    cloudinary.config({
      cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
      api_key: process.env.CLOUDINARY_API_KEY,
      api_secret: process.env.CLOUDINARY_API_SECRET,
    });
  }

  async uploadFile(
    file: Express.Multer.File,
  ): Promise<{ url: string; type: string }> {
    this.validateFile(file);
    const isVideo = file.mimetype.startsWith('video/');

    return new Promise((resolve, reject) => {
      const uploadOptions = {
        resource_type: isVideo ? 'video' : ('image' as 'video' | 'image'),
      };

      cloudinary.uploader
        .upload_stream(uploadOptions, (error, result) => {
          if (error) return reject(error);
          resolve({
            url: result.secure_url,
            type: result.resource_type,
          });
        })
        .end(file.buffer);
    });
  }

  async uploadMultipleFiles(
    files: Express.Multer.File[],
  ): Promise<{ url: string; type: string }[]> {
    return await Promise.all(
      files.map((file) => {
        this.validateFile(file);

        return this.uploadFile(file);
      }),
    );
  }

  validateFile(file: Express.Multer.File) {
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      throw new BadRequestException('File size exceeds the limit (5MB)');
    }

    // File type validation
    const allowedMimeTypes = [
      'image/jpeg',
      'image/png',
      'image/gif',
      'image/jpg',
      'video/mp4',
    ];

    if (!allowedMimeTypes.includes(file.mimetype)) {
      throw new BadRequestException(
        'Invalid file type. Only JPEG, PNG, JPG, GIF, and MP4 are allowed.',
      );
    }
  }
}
