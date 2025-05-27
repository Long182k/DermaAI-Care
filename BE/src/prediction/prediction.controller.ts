import {
  Controller,
  HttpException,
  HttpStatus,
  Post,
  UploadedFile,
  UseInterceptors,
  Get,
  Param,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiConsumes, ApiTags } from '@nestjs/swagger';
import { CurrentUser } from 'src/auth/@decorator/current-user.decorator';
import { PredictionService } from './prediction.service';

@ApiTags('prediction')
@Controller('prediction')
export class PredictionController {
  constructor(private readonly predictionService: PredictionService) {}

  @Post('detect')
  @ApiConsumes('multipart/form-data')
  @UseInterceptors(FileInterceptor('image'))
  async detectLesion(
    @CurrentUser() user: any,
    @UploadedFile() file: Express.Multer.File,
  ) {
    if (!file) {
      throw new HttpException('Image file is required', HttpStatus.BAD_REQUEST);
    }
    const userId = user.id ?? user.userId;
    return this.predictionService.predict(userId, file);
  }

  @Get('history')
  async getUserPredictionHistory(@CurrentUser() user: any) {
    const userId = user.id ?? user.userId;
    return this.predictionService.getUserPredictionHistory(userId);
  }

  @Get('doctor/history/:userId')
  async doctorGetUserPredictionHistory(@Param('userId') userId: string) {
    return this.predictionService.getUserPredictionHistory(userId);
  }
}
