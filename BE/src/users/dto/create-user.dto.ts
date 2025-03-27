import { Languages, Role, Gender } from '@prisma/client';
import { IsEmail, IsNotEmpty, IsOptional, IsString } from 'class-validator';

export class CreateUserDTO {
  @IsString()
  @IsNotEmpty()
  userName: string;

  @IsString()
  @IsNotEmpty()
  firstName: string;

  @IsString()
  @IsNotEmpty()
  lastName: string;

  @IsString()
  @IsNotEmpty()
  role: Role;

  @IsString()
  @IsNotEmpty()
  password: string;

  @IsEmail()
  @IsNotEmpty()
  email: string;

  @IsString()
  @IsOptional()
  avatarUrl?: string;

  @IsOptional()
  dateOfBirth?: Date;

  @IsOptional()
  experience?: number;

  @IsString()
  @IsOptional()
  education?: string;

  @IsString()
  @IsOptional()
  certifications?: string;

  @IsOptional()
  languages?: Languages;

  @IsString()
  @IsOptional()
  phoneNumber?: string;

  @IsOptional()
  gender?: Gender;
}
