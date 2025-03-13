import { Module } from '@nestjs/common';
import { ScheduleController } from './schedule.controller';
import { ScheduleService } from './schedule.service';
import { PrismaService } from '../prisma.service';
import { ScheduleTasksService } from './schedule-tasks.service';
import { JwtService } from '@nestjs/jwt';

@Module({
  controllers: [ScheduleController],
  providers: [ScheduleService, PrismaService, ScheduleTasksService, JwtService],
  exports: [ScheduleService],
})
export class ScheduleModule {}
