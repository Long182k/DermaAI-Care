import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { Role } from '@prisma/client';
import { PrismaService } from '../prisma.service';
import { ScheduleService } from './schedule.service';

@Injectable()
export class ScheduleTasksService {
  private readonly logger = new Logger(ScheduleTasksService.name);

  constructor(
    private prisma: PrismaService,
    private scheduleService: ScheduleService,
  ) {}

  // Run at midnight on the first day of every month
  @Cron('0 0 1 * *')
  async handleMonthlyScheduleGeneration() {
    this.logger.log('Starting monthly schedule generation for doctors...');
    await this.generateSchedulesForDoctors();
  }

  async generateSchedulesForDoctors() {
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
        this.logger.log(
          `Generated monthly schedule for doctor: ${doctor.userName}`,
        );
      }

      this.logger.log('Monthly schedule generation completed successfully');
      return { message: 'Schedule generation completed successfully' };
    } catch (error) {
      this.logger.error('Error during schedule generation:', error.stack);
      throw error;
    }
  }
}
