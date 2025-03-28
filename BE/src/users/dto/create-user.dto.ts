import { Languages, Role } from '@prisma/client';
import { IsEmail, IsNotEmpty, IsOptional, IsString } from 'class-validator';

export class CreateUserDTO {
  @IsString()
  @IsNotEmpty()
  userName: string;

  @IsString()
  @IsNotEmpty()
  role: Role;

  @IsString()
  @IsNotEmpty()
  password: string;

  @IsEmail()
  @IsNotEmpty()
  email: string;

  @IsOptional()
  avatarUrl?: string;

  @IsOptional()
  dateOfBirth?: Date;

  @IsOptional()
  experience?: number;

  @IsOptional()
  education?: string;

  @IsOptional()
  certifications?: string;

  @IsOptional()
  languages?: Languages;
}
