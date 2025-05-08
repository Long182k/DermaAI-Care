import { Module } from '@nestjs/common';
import { PredictionService } from './prediction.service';
import { PredictionController } from './prediction.controller';
import { PrismaService } from 'src/prisma.service';
import { CloudinaryService } from 'src/file/file.service';

@Module({
  controllers: [PredictionController],
  providers: [PredictionService, PrismaService, CloudinaryService],
})
export class PredictionModule {}
