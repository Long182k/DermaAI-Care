import { IsEmail, IsNotEmpty, IsOptional, IsString } from 'class-validator';

export class GetUserByKeywordDTO {
  @IsString()
  @IsOptional()
  userName?: string;

  @IsString()
  @IsOptional()
  id?: string;

  @IsEmail()
  @IsOptional()
  email?: string;

  @IsString()
  @IsOptional()
  avatarUrl?: string;

  @IsString()
  @IsOptional()
  bio?: string;
}
