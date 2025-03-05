import { IsDate, IsEnum, IsISO8601, IsOptional } from 'class-validator';
import { ScheduleStatus } from '@prisma/client';

export class UpdateScheduleDto {
  @IsOptional()
  @IsISO8601()
  startTime?: string;

  @IsOptional()
  @IsISO8601()
  endTime?: string;

  @IsOptional()
  @IsEnum(ScheduleStatus)
  status?: ScheduleStatus;
}
