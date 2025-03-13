import { IsDate, IsEnum, IsISO8601, IsOptional } from 'class-validator';
import { ScheduleStatus } from '@prisma/client';

export class UpdateScheduleDto {
  @IsOptional()
  startTime?: string;

  @IsOptional()
  endTime?: string;

  @IsOptional()
  @IsEnum(ScheduleStatus)
  status?: ScheduleStatus;
}
