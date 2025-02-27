import {
  Controller,
  Post,
  UseInterceptors,
  UploadedFile,
  UploadedFiles,
  BadRequestException,
} from '@nestjs/common';
import { FileInterceptor, FilesInterceptor } from '@nestjs/platform-express';
import { CloudinaryService } from './file.service';

@Controller('upload')
export class FileUploadController {
  constructor(private readonly cloudinaryService: CloudinaryService) {}

  @Post('single')
  @UseInterceptors(FileInterceptor('file'))
  async uploadSingleFile(@UploadedFile() file: Express.Multer.File) {
    if (!file) {
      throw new BadRequestException('No file uploaded');
    }

    const uploadedUrl = await this.cloudinaryService.uploadFile(file);
    return { url: uploadedUrl };
  }

  @Post('multiple')
  @UseInterceptors(FilesInterceptor('files', 5))
  async uploadMultipleFiles(@UploadedFiles() files: Express.Multer.File[]) {
    if (!files || files.length === 0) {
      throw new BadRequestException('No files uploaded');
    }

    const uploadedUrls =
      await this.cloudinaryService.uploadMultipleFiles(files);

    return { urls: uploadedUrls };
  }
}
