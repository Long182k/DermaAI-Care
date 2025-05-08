import { ApiProperty } from '@nestjs/swagger';

export class CreatePredictionDto {
  @ApiProperty({ type: 'string', format: 'binary', description: 'Skin lesion image to analyze' })
  image: Express.Multer.File;
}