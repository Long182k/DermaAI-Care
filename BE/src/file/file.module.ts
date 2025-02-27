import { Module } from '@nestjs/common';
import { CloudinaryService } from './file.service';

@Module({
  providers: [CloudinaryService],
  exports: [CloudinaryService],
})
export class FileModule {}
