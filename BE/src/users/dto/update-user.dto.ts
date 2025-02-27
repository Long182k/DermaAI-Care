import {
  IsDateString,
  IsNotEmpty,
  IsOptional,
  IsString,
  Length,
} from 'class-validator';

export class UpdateUserDto {
  @IsOptional()
  @IsString()
  @Length(3, 20)
  userName?: string;

  @IsOptional()
  @IsString()
  @Length(0, 500)
  bio?: string;

  @IsOptional()
  @IsDateString()
  dateOfBirth?: string;
}

export class UpdateHashedRefreshTokenDTO {
  @IsString()
  @IsNotEmpty()
  hashedRefreshToken: string;

  @IsString()
  @IsNotEmpty()
  userId: string;
}
