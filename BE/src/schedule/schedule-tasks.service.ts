import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PrismaService } from '../prisma.service';
import { ScheduleService } from './schedule.service';
import { Role } from '@prisma/client';

@Injectable()
export class ScheduleTasksService {
  private readonly logger = new Logger(ScheduleTasksService.name);

  constructor(
    private prisma: PrismaService,
    private scheduleService: ScheduleService,
  ) {}

  // Run at 00:00 on January 1st
  @Cron('0 0 1 1 *')
  async handleYearlyScheduleGeneration() {
    this.logger.log('Starting yearly schedule generation for doctors...');

    try {
      // Get all doctors
      const doctors = await this.prisma.user.findMany({
        where: {
          role: Role.DOCTOR,
          isActive: true,
        },
      });

      // Generate schedules for each doctor
      for (const doctor of doctors) {
        await this.scheduleService.generateDefaultSchedule(doctor.id);
        this.logger.log(`Generated schedule for doctor: ${doctor.userName}`);
      }

      this.logger.log('Yearly schedule generation completed successfully');
    } catch (error) {
      this.logger.error(
        'Error during yearly schedule generation:',
        error.stack,
      );
    }
  }
}
