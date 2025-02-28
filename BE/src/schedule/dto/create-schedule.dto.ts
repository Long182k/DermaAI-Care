import { IsDate, IsEnum, IsString } from 'class-validator';
import { ScheduleStatus } from '@prisma/client';

export class CreateScheduleDto {
  @IsString()
  doctorId: string;

  @IsDate()
  startTime: Date;

  @IsDate()
  endTime: Date;

  @IsEnum(ScheduleStatus)
  status: ScheduleStatus;
}
