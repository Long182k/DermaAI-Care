import { IsDate, IsEnum, IsISO8601, IsString } from 'class-validator';
import { ScheduleStatus } from '@prisma/client';

export class CreateScheduleDto {
  @IsString()
  doctorId: string;

  @IsISO8601()
  startTime: string;

  @IsISO8601()
  endTime: string;

  @IsEnum(ScheduleStatus)
  status: ScheduleStatus;
}
