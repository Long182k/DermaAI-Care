import { IsString, IsOptional } from 'class-validator';

export class CreateAppointmentDto {
  @IsString()
  scheduleId: string;

  @IsOptional()
  @IsString()
  notes?: string;

  @IsOptional()
  @IsString()
  patientId?: string;
}
